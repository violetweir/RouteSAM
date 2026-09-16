@torch.inference_mode()
def add_tracker_exact_mask(
    self,
    inference_state,
    frame_idx,
    obj_id,
    mask,
    rel_coordinates=True,
    use_prev_mem_frame=False,
):
    """Add a new point prompt to Tracker. Suppporting instance refinement to existing
    objects by passing existing obj_id or adding a new object by passing a new obj_id.
    use_prev_mem_frame=False to disable cross attention to previous memory frames.
    Every GPU returns the same results, and results should contain all masks including
    these masks not refined or not added by the current user points.
    """
    assert obj_id is not None, "obj_id must be provided to add new points"
    tracker_metadata = inference_state["tracker_metadata"]
    if tracker_metadata == {}:
        # initialize masklet metadata if it's uninitialized (empty dict)
        tracker_metadata.update(self._initialize_metadata())

    obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)

    # prepare feature
    self._prepare_backbone_feats(inference_state, frame_idx, reverse=False)

    object_has_been_refined = self._has_object_been_refined(inference_state, obj_id)
    if (
        obj_rank is not None
        and self.use_stateless_refinement
        and not object_has_been_refined
    ):
        # The first time we start refinement on the object, we remove it.
        logger.debug(
            f"[rank={self.rank}] Removing object {obj_id} before refinement."
        )
        self.remove_object(inference_state, obj_id, is_user_action=False)
        obj_rank = None

    if obj_rank is None:
        # new object, we assign it a GPU and create a new inference state if limit allows
        num_prev_obj = np.sum(tracker_metadata["num_obj_per_gpu"])
        if num_prev_obj >= self.max_num_objects:
            logger.warning(
                f"add_tracker_new_points: cannot add a new object as we are already tracking {num_prev_obj=} "
                f"masklets (under {self.max_num_objects=})"
            )
            obj_ids = []
            H_low_res = W_low_res = self.tracker.low_res_mask_size
            H_video_res = inference_state["orig_height"]
            W_video_res = inference_state["orig_width"]
            low_res_masks = torch.zeros(0, 1, H_low_res, W_low_res)
            video_res_masks = torch.zeros(0, 1, H_video_res, W_video_res)
            return frame_idx, obj_ids, low_res_masks, video_res_masks

        new_det_gpu_ids = self._assign_new_det_to_gpus(
            new_det_num=1,
            prev_workload_per_gpu=tracker_metadata["num_obj_per_gpu"],
        )
        obj_rank = new_det_gpu_ids[0]

        # get tracker inference state for the new object
        if self.rank == obj_rank:
            # for batched inference, we create a new inference state
            tracker_state = self._init_new_tracker_state(inference_state)
            inference_state["tracker_inference_states"].append(tracker_state)

        # update metadata
        tracker_metadata["obj_ids_per_gpu"][obj_rank] = np.concatenate(
            [
                tracker_metadata["obj_ids_per_gpu"][obj_rank],
                np.array([obj_id], dtype=np.int64),
            ]
        )
        tracker_metadata["num_obj_per_gpu"][obj_rank] = len(
            tracker_metadata["obj_ids_per_gpu"][obj_rank]
        )
        tracker_metadata["obj_ids_all_gpu"] = np.concatenate(
            tracker_metadata["obj_ids_per_gpu"]
        )
        tracker_metadata["max_obj_id"] = max(tracker_metadata["max_obj_id"], obj_id)

        logger.debug(
            f"[rank={self.rank}] Adding new object with id {obj_id} at frame {frame_idx}."
        )
        self.add_action_history(
            inference_state, "add", frame_idx=frame_idx, obj_ids=[obj_id]
        )
    else:
        # existing object, for refinement
        if self.rank == obj_rank:
            tracker_states = self._get_tracker_inference_states_by_obj_ids(
                inference_state, [obj_id]
            )
            assert len(tracker_states) == 1, (
                f"[rank={self.rank}] Multiple Tracker inference states found for the same object id."
            )
            tracker_state = tracker_states[0]

        # log
        logger.debug(
            f"[rank={self.rank}] Refining existing object with id {obj_id} at frame {frame_idx}."
        )
        self.add_action_history(
            inference_state, "refine", frame_idx=frame_idx, obj_ids=[obj_id]
        )

    # assign higher score to added/refined object
    tracker_metadata["obj_id_to_score"][obj_id] = 1.0
    tracker_metadata["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = 1.0

    if self.rank == 0:
        rank0_metadata = tracker_metadata.get("rank0_metadata", {})

        if "removed_obj_ids" in rank0_metadata:
            rank0_metadata["removed_obj_ids"].discard(obj_id)

        if "suppressed_obj_ids" in rank0_metadata:
            for frame_id in rank0_metadata["suppressed_obj_ids"]:
                rank0_metadata["suppressed_obj_ids"][frame_id].discard(obj_id)

        if "masklet_confirmation" in rank0_metadata:
            obj_ids_all_gpu = tracker_metadata["obj_ids_all_gpu"]
            obj_indices = np.where(obj_ids_all_gpu == obj_id)[0]
            if len(obj_indices) > 0:
                obj_idx = obj_indices[0]
                if obj_idx < len(rank0_metadata["masklet_confirmation"]["status"]):
                    rank0_metadata["masklet_confirmation"]["status"][obj_idx] = 1
                    rank0_metadata["masklet_confirmation"]["consecutive_det_num"][
                        obj_idx
                    ] = self.masklet_confirmation_consecutive_det_thresh

    if self.rank == obj_rank:
        frame_idx, obj_ids, low_res_masks, video_res_masks = (
            self.tracker.add_new_mask(
                inference_state=tracker_state, frame_idx=frame_idx, obj_id=obj_id,
                mask=mask, add_mask_to_memory=True,
            )
        )

        if video_res_masks is not None and len(video_res_masks) > 0:
            video_res_masks = fill_holes_in_mask_scores(
                video_res_masks,  # shape (N, 1, H_video, W_video)
                max_area=self.fill_hole_area,
                fill_holes=True,
                remove_sprinkles=True,
            )

        # Since the mem encoder has already run for the current input points?
        self.tracker.propagate_in_video_preflight(
            tracker_state, run_mem_encoder=True
        )
        # Clear detector conditioning frames when user clicks are received to allow
        # model updating masks on these frames. It is a noop if user is refining on the
        # detector conditioning frames or adding new objects.
        # Preserve user mask conditioning frame.

    # fetch results from states and gather across GPUs
    # Use optimized caching approach to avoid reprocessing unmodified objects
    if self.rank == obj_rank and len(obj_ids) > 0:
        new_mask_data = (video_res_masks[obj_ids.index(obj_id)] > 0.0).to(
            torch.bool
        )
    else:
        new_mask_data = None
    # Broadcast the new mask data across all ranks for consistency
    if self.world_size > 1:
        data_list = [new_mask_data.cpu() if new_mask_data is not None else None]
        self.broadcast_python_obj_cpu(data_list, src=obj_rank)
        new_mask_data = data_list[0].to(self.device)

    if self.rank == 0:
        obj_id_to_mask = self._build_tracker_output(
            inference_state,
            frame_idx,
            {obj_id: new_mask_data} if new_mask_data is not None else None,
        )
        # post processing - remove suppressed obj_ids
        obj_id_to_score = tracker_metadata["obj_id_to_score"]
        suppressed_obj_ids = tracker_metadata["rank0_metadata"][
            "suppressed_obj_ids"
        ][frame_idx]
        obj_id_to_tracker_score = tracker_metadata[
            "obj_id_to_tracker_score_frame_wise"
        ][frame_idx]

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,
            "obj_id_to_tracker_score": obj_id_to_tracker_score,
        }
        self._cache_frame_outputs(
            inference_state,
            frame_idx,
            obj_id_to_mask,
            suppressed_obj_ids=suppressed_obj_ids,
        )
        return frame_idx, self._postprocess_output(
            inference_state, out, suppressed_obj_ids=suppressed_obj_ids
        )
    else:
        return frame_idx, None  # no output on other GPUs
