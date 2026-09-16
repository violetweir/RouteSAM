import sys,runpy,torch
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.25)
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
