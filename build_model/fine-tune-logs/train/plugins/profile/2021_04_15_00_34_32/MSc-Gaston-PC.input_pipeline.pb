$	怙>�g�?�0���V�?��gB��r?!�YL�?	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails]�6��n�?�-c}�?A�5v�ꭡ?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��x@�t?�Go���j?A߿yq�]?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailss�69|ҵ? a��*v?Aa�4��o�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailshz��L��?�w}�O�?A�8���֬?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��gB��r?�'��Ql?A;�O��nR?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsm�Yg|�?�\��?AҧU�fn?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails���%v?���m3U?A]N	�I�p?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails"��3�c�?��h��k?Ad����~?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsS=��Mz?�~j�t�X?A��~P)t?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	�YL�?]��e�?Aan�r��?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
aU��N��?����?A�6�ُa?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�б�J�?�
�+�j�?A���� \?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!yv�֧?���k�˦?A�7�0�`?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�R���Ɨ?6;R}��?Af�ʉve?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��ٮ��?[��X��?A76;R}�W?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails,I���p�?�����?Alxz�,C\?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailszZ����?��ӀAҧ?A�ɐc�Y?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsfh<�y�?�@�mߣ�?Ak~��E}b?*	�C�l��R@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat/ޏ�/��?!�"*���B@)�R���җ?1т��>@:Preprocessing2F
Iterator::ModelZh�4��?!�7ě��A@)�Y��U��?1���o�5@:Preprocessing2U
Iterator::Model::ParallelMapV2;�zj��?!��P7?m,@);�zj��?1��P7?m,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�mR�X��?!��Ȯ2@)#K�Xޅ?1��2bO,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�n��Ũ?!�2�P@)LOX�es?1!��|@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�-v��2s?!1�E�T�@)�-v��2s?11�E�T�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�)��sh?!̷T���@)�)��sh?1̷T���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�I����?!�pI��4@)p�h���`?1�`\�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 55.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	pܞom�?�j�>�?���m3U?!�
�+�j�?	!       "	!       *	!       2$	đV��a�?t�!��]�?;�O��nR?!an�r��?:	!       B	!       J	!       R	!       Z	!       JCPU_ONLYb 