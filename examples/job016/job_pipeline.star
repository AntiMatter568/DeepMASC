
# version 30001

data_pipeline_general

_rlnPipeLineJobCounter                       2
 

# version 30001

data_pipeline_processes

loop_ 
_rlnPipeLineProcessName #1 
_rlnPipeLineProcessAlias #2 
_rlnPipeLineProcessTypeLabel #3 
_rlnPipeLineProcessStatusLabel #4 
Class3D/job016/       None relion.class3d    Running 
 

# version 30001

data_pipeline_nodes

loop_ 
_rlnPipeLineNodeName #1 
_rlnPipeLineNodeTypeLabel #2 
Select/job014/particles.star ParticlesData.star.relion 
InitialModel/job015/initial_model.mrc DensityMap.mrc 
Class3D/job016/run_it025_data.star ParticlesData.star.relion.refine3d 
Class3D/job016/run_it025_optimiser.star ProcessData.star.relion.optimiser.class3d 
Class3D/job016/run_it025_class001.mrc DensityMap.mrc.relion.class3d 
Class3D/job016/run_it025_class002.mrc DensityMap.mrc.relion.class3d 
Class3D/job016/run_it025_class003.mrc DensityMap.mrc.relion.class3d 
Class3D/job016/run_it025_class004.mrc DensityMap.mrc.relion.class3d 
 

# version 30001

data_pipeline_input_edges

loop_ 
_rlnPipeLineEdgeFromNode #1 
_rlnPipeLineEdgeProcess #2 
Select/job014/particles.star Class3D/job016/ 
InitialModel/job015/initial_model.mrc Class3D/job016/ 
 

# version 30001

data_pipeline_output_edges

loop_ 
_rlnPipeLineEdgeProcess #1 
_rlnPipeLineEdgeToNode #2 
Class3D/job016/ Class3D/job016/run_it025_data.star 
Class3D/job016/ Class3D/job016/run_it025_optimiser.star 
Class3D/job016/ Class3D/job016/run_it025_class001.mrc 
Class3D/job016/ Class3D/job016/run_it025_class002.mrc 
Class3D/job016/ Class3D/job016/run_it025_class003.mrc 
Class3D/job016/ Class3D/job016/run_it025_class004.mrc 
 
