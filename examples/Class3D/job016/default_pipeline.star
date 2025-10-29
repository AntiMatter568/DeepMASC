
# version 30001

data_pipeline_general

_rlnPipeLineJobCounter                      17
 

# version 30001

data_pipeline_processes

loop_ 
_rlnPipeLineProcessName #1 
_rlnPipeLineProcessAlias #2 
_rlnPipeLineProcessTypeLabel #3 
_rlnPipeLineProcessStatusLabel #4 
Import/job001/       None relion.import.movies  Succeeded 
MotionCorr/job002/       None relion.motioncorr.own  Succeeded 
CtfFind/job003/       None relion.ctffind.ctffind4  Succeeded 
ManualPick/job004/       None relion.manualpick  Succeeded 
Select/job005/       None relion.select.split  Succeeded 
AutoPick/job006/       None relion.autopick.log  Succeeded 
Extract/job007/       None relion.extract  Succeeded 
Class2D/job008/       None relion.class2d  Succeeded 
Select/job009/       None relion.select.class2dauto  Succeeded 
AutoPick/job010/       None relion.autopick.topaz.train  Succeeded 
AutoPick/job011/       None relion.autopick.topaz.pick  Succeeded 
Extract/job012/       None relion.extract  Succeeded 
Class2D/job013/       None relion.class2d  Succeeded 
Select/job014/       None relion.select.class2dauto  Succeeded 
InitialModel/job015/       None relion.initialmodel  Succeeded 
Select/job017/       None relion.select.interactive  Succeeded 
Class3D/job016/       None relion.class3d    Running 
 

# version 30001

data_pipeline_nodes

loop_ 
_rlnPipeLineNodeName #1 
_rlnPipeLineNodeTypeLabel #2 
Import/job001/movies.star MicrographMoviesData.star.relion 
MotionCorr/job002/corrected_micrographs.star MicrographsData.star.relion.motioncorr 
MotionCorr/job002/logfile.pdf LogFile.pdf.relion.motioncorr 
CtfFind/job003/micrographs_ctf.star MicrographsData.star.relion.ctf 
CtfFind/job003/logfile.pdf LogFile.pdf.relion.ctffind 
ManualPick/job004/micrographs_selected.star MicrographsData.star.relion 
ManualPick/job004/manualpick.star MicrographsCoords.star.relion.manualpick 
Select/job005/micrographs_split1.star MicrographsData.star.relion 
Select/job005/micrographs_split2.star MicrographsData.star.relion 
Select/job005/micrographs_split3.star MicrographsData.star.relion 
AutoPick/job006/autopick.star MicrographsCoords.star.relion.autopick 
AutoPick/job006/logfile.pdf LogFile.pdf.relion.autopick 
Extract/job007/particles.star ParticlesData.star.relion 
Class2D/job008/run_it025_data.star ParticlesData.star.relion.class2d 
Class2D/job008/run_it025_optimiser.star ProcessData.star.relion.optimiser.class2d 
Select/job009/particles.star ParticlesData.star.relion 
Select/job009/class_averages.star ImagesData.star.relion.classaverages 
Select/job009/rank_optimiser.star ProcessData.star.relion.optimiser.select 
AutoPick/job010/input_training_coords.star MicrographsCoords.star.relion 
AutoPick/job011/autopick.star MicrographsCoords.star.relion.autopick 
AutoPick/job011/logfile.pdf LogFile.pdf.relion.autopick 
Extract/job012/particles.star ParticlesData.star.relion 
Class2D/job013/run_it100_data.star ParticlesData.star.relion.class2d 
Class2D/job013/run_it100_optimiser.star ProcessData.star.relion.optimiser.class2d 
Select/job014/particles.star ParticlesData.star.relion 
Select/job014/class_averages.star ImagesData.star.relion.classaverages 
Select/job014/rank_optimiser.star ProcessData.star.relion.optimiser.select 
InitialModel/job015/initial_model.mrc DensityMap.mrc.relion.initialmodel 
Select/job017/particles.star ParticlesData.star.relion 
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
Import/job001/movies.star MotionCorr/job002/ 
MotionCorr/job002/corrected_micrographs.star CtfFind/job003/ 
CtfFind/job003/micrographs_ctf.star ManualPick/job004/ 
CtfFind/job003/micrographs_ctf.star Select/job005/ 
Select/job005/micrographs_split1.star AutoPick/job006/ 
CtfFind/job003/micrographs_ctf.star Extract/job007/ 
AutoPick/job006/autopick.star Extract/job007/ 
Extract/job007/particles.star Class2D/job008/ 
Class2D/job008/run_it025_optimiser.star Select/job009/ 
Select/job005/micrographs_split1.star AutoPick/job010/ 
CtfFind/job003/micrographs_ctf.star AutoPick/job011/ 
CtfFind/job003/micrographs_ctf.star Extract/job012/ 
AutoPick/job011/autopick.star Extract/job012/ 
Extract/job012/particles.star Class2D/job013/ 
Class2D/job013/run_it100_optimiser.star Select/job014/ 
Select/job014/particles.star InitialModel/job015/ 
Select/job014/particles.star Class3D/job016/ 
InitialModel/job015/initial_model.mrc Class3D/job016/ 
 

# version 30001

data_pipeline_output_edges

loop_ 
_rlnPipeLineEdgeProcess #1 
_rlnPipeLineEdgeToNode #2 
Import/job001/ Import/job001/movies.star 
MotionCorr/job002/ MotionCorr/job002/corrected_micrographs.star 
MotionCorr/job002/ MotionCorr/job002/logfile.pdf 
CtfFind/job003/ CtfFind/job003/micrographs_ctf.star 
CtfFind/job003/ CtfFind/job003/logfile.pdf 
ManualPick/job004/ ManualPick/job004/micrographs_selected.star 
ManualPick/job004/ ManualPick/job004/manualpick.star 
Select/job005/ Select/job005/micrographs_split1.star 
Select/job005/ Select/job005/micrographs_split2.star 
Select/job005/ Select/job005/micrographs_split3.star 
AutoPick/job006/ AutoPick/job006/autopick.star 
AutoPick/job006/ AutoPick/job006/logfile.pdf 
Extract/job007/ Extract/job007/particles.star 
Class2D/job008/ Class2D/job008/run_it025_data.star 
Class2D/job008/ Class2D/job008/run_it025_optimiser.star 
Select/job009/ Select/job009/particles.star 
Select/job009/ Select/job009/class_averages.star 
Select/job009/ Select/job009/rank_optimiser.star 
AutoPick/job010/ AutoPick/job010/input_training_coords.star 
AutoPick/job011/ AutoPick/job011/autopick.star 
AutoPick/job011/ AutoPick/job011/logfile.pdf 
Extract/job012/ Extract/job012/particles.star 
Class2D/job013/ Class2D/job013/run_it100_data.star 
Class2D/job013/ Class2D/job013/run_it100_optimiser.star 
Select/job014/ Select/job014/particles.star 
Select/job014/ Select/job014/class_averages.star 
Select/job014/ Select/job014/rank_optimiser.star 
InitialModel/job015/ InitialModel/job015/initial_model.mrc 
Select/job017/ Select/job017/particles.star 
Class3D/job016/ Class3D/job016/run_it025_data.star 
Class3D/job016/ Class3D/job016/run_it025_optimiser.star 
Class3D/job016/ Class3D/job016/run_it025_class001.mrc 
Class3D/job016/ Class3D/job016/run_it025_class002.mrc 
Class3D/job016/ Class3D/job016/run_it025_class003.mrc 
Class3D/job016/ Class3D/job016/run_it025_class004.mrc 
 
