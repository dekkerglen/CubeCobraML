��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
�
draft_reconstruction/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:��**
shared_namedraft_reconstruction/bias
�
-draft_reconstruction/bias/Read/ReadVariableOpReadVariableOpdraft_reconstruction/bias*
_output_shapes

:��*
dtype0
�
draft_reconstruction/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*,
shared_namedraft_reconstruction/kernel
�
/draft_reconstruction/kernel/Read/ReadVariableOpReadVariableOpdraft_reconstruction/kernel*!
_output_shapes
:���*
dtype0
s
draft_d3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedraft_d3/bias
l
!draft_d3/bias/Read/ReadVariableOpReadVariableOpdraft_d3/bias*
_output_shapes	
:�*
dtype0
|
draft_d3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedraft_d3/kernel
u
#draft_d3/kernel/Read/ReadVariableOpReadVariableOpdraft_d3/kernel* 
_output_shapes
:
��*
dtype0
s
draft_d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedraft_d1/bias
l
!draft_d1/bias/Read/ReadVariableOpReadVariableOpdraft_d1/bias*
_output_shapes	
:�*
dtype0
|
draft_d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedraft_d1/kernel
u
#draft_d1/kernel/Read/ReadVariableOpReadVariableOpdraft_d1/kernel* 
_output_shapes
:
��*
dtype0
�
serving_default_draft_d1_inputPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_draft_d1_inputdraft_d1/kerneldraft_d1/biasdraft_d3/kerneldraft_d3/biasdraft_reconstruction/kerneldraft_reconstruction/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_54058

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias*
.
0
1
2
3
"4
#5*
.
0
1
2
3
"4
#5*
* 
�
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
)trace_0
*trace_1
+trace_2
,trace_3* 
6
-trace_0
.trace_1
/trace_2
0trace_3* 
* 

1serving_default* 

0
1*

0
1*
* 
�
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

7trace_0* 

8trace_0* 
_Y
VARIABLE_VALUEdraft_d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdraft_d1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

>trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdraft_d3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdraft_d3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

"0
#1*

"0
#1*
* 
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

Etrace_0* 

Ftrace_0* 
ke
VARIABLE_VALUEdraft_reconstruction/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEdraft_reconstruction/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedraft_d1/kerneldraft_d1/biasdraft_d3/kerneldraft_d3/biasdraft_reconstruction/kerneldraft_reconstruction/biasConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_54261
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedraft_d1/kerneldraft_d1/biasdraft_d3/kerneldraft_d3/biasdraft_reconstruction/kerneldraft_reconstruction/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_54289��
�%
�
 __inference__wrapped_model_53827
draft_d1_inputH
4sequential_2_draft_d1_matmul_readvariableop_resource:
��D
5sequential_2_draft_d1_biasadd_readvariableop_resource:	�H
4sequential_2_draft_d3_matmul_readvariableop_resource:
��D
5sequential_2_draft_d3_biasadd_readvariableop_resource:	�U
@sequential_2_draft_reconstruction_matmul_readvariableop_resource:���Q
Asequential_2_draft_reconstruction_biasadd_readvariableop_resource:
��
identity��,sequential_2/draft_d1/BiasAdd/ReadVariableOp�+sequential_2/draft_d1/MatMul/ReadVariableOp�,sequential_2/draft_d3/BiasAdd/ReadVariableOp�+sequential_2/draft_d3/MatMul/ReadVariableOp�8sequential_2/draft_reconstruction/BiasAdd/ReadVariableOp�7sequential_2/draft_reconstruction/MatMul/ReadVariableOp�
+sequential_2/draft_d1/MatMul/ReadVariableOpReadVariableOp4sequential_2_draft_d1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_2/draft_d1/MatMulMatMuldraft_d1_input3sequential_2/draft_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_2/draft_d1/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_draft_d1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_2/draft_d1/BiasAddBiasAdd&sequential_2/draft_d1/MatMul:product:04sequential_2/draft_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_2/draft_d1/ReluRelu&sequential_2/draft_d1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
+sequential_2/draft_d3/MatMul/ReadVariableOpReadVariableOp4sequential_2_draft_d3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_2/draft_d3/MatMulMatMul(sequential_2/draft_d1/Relu:activations:03sequential_2/draft_d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_2/draft_d3/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_draft_d3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_2/draft_d3/BiasAddBiasAdd&sequential_2/draft_d3/MatMul:product:04sequential_2/draft_d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
sequential_2/draft_d3/ReluRelu&sequential_2/draft_d3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
7sequential_2/draft_reconstruction/MatMul/ReadVariableOpReadVariableOp@sequential_2_draft_reconstruction_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
(sequential_2/draft_reconstruction/MatMulMatMul(sequential_2/draft_d3/Relu:activations:0?sequential_2/draft_reconstruction/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
8sequential_2/draft_reconstruction/BiasAdd/ReadVariableOpReadVariableOpAsequential_2_draft_reconstruction_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
)sequential_2/draft_reconstruction/BiasAddBiasAdd2sequential_2/draft_reconstruction/MatMul:product:0@sequential_2/draft_reconstruction/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
)sequential_2/draft_reconstruction/SigmoidSigmoid2sequential_2/draft_reconstruction/BiasAdd:output:0*
T0*)
_output_shapes
:�����������~
IdentityIdentity-sequential_2/draft_reconstruction/Sigmoid:y:0^NoOp*
T0*)
_output_shapes
:������������
NoOpNoOp-^sequential_2/draft_d1/BiasAdd/ReadVariableOp,^sequential_2/draft_d1/MatMul/ReadVariableOp-^sequential_2/draft_d3/BiasAdd/ReadVariableOp,^sequential_2/draft_d3/MatMul/ReadVariableOp9^sequential_2/draft_reconstruction/BiasAdd/ReadVariableOp8^sequential_2/draft_reconstruction/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2\
,sequential_2/draft_d1/BiasAdd/ReadVariableOp,sequential_2/draft_d1/BiasAdd/ReadVariableOp2Z
+sequential_2/draft_d1/MatMul/ReadVariableOp+sequential_2/draft_d1/MatMul/ReadVariableOp2\
,sequential_2/draft_d3/BiasAdd/ReadVariableOp,sequential_2/draft_d3/BiasAdd/ReadVariableOp2Z
+sequential_2/draft_d3/MatMul/ReadVariableOp+sequential_2/draft_d3/MatMul/ReadVariableOp2t
8sequential_2/draft_reconstruction/BiasAdd/ReadVariableOp8sequential_2/draft_reconstruction/BiasAdd/ReadVariableOp2r
7sequential_2/draft_reconstruction/MatMul/ReadVariableOp7sequential_2/draft_reconstruction/MatMul/ReadVariableOp:X T
(
_output_shapes
:����������
(
_user_specified_namedraft_d1_input
�

�
O__inference_draft_reconstruction_layer_call_and_return_conditional_losses_53876

inputs3
matmul_readvariableop_resource:���/
biasadd_readvariableop_resource:
��
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0k
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������t
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:��*
dtype0x
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������X
SigmoidSigmoidBiasAdd:output:0*
T0*)
_output_shapes
:�����������\
IdentityIdentitySigmoid:y:0^NoOp*
T0*)
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_sequential_2_layer_call_and_return_conditional_losses_53902
draft_d1_input"
draft_d1_53886:
��
draft_d1_53888:	�"
draft_d3_53891:
��
draft_d3_53893:	�/
draft_reconstruction_53896:���*
draft_reconstruction_53898:
��
identity�� draft_d1/StatefulPartitionedCall� draft_d3/StatefulPartitionedCall�,draft_reconstruction/StatefulPartitionedCall�
 draft_d1/StatefulPartitionedCallStatefulPartitionedCalldraft_d1_inputdraft_d1_53886draft_d1_53888*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_draft_d1_layer_call_and_return_conditional_losses_53842�
 draft_d3/StatefulPartitionedCallStatefulPartitionedCall)draft_d1/StatefulPartitionedCall:output:0draft_d3_53891draft_d3_53893*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_draft_d3_layer_call_and_return_conditional_losses_53859�
,draft_reconstruction/StatefulPartitionedCallStatefulPartitionedCall)draft_d3/StatefulPartitionedCall:output:0draft_reconstruction_53896draft_reconstruction_53898*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_draft_reconstruction_layer_call_and_return_conditional_losses_53876�
IdentityIdentity5draft_reconstruction/StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:������������
NoOpNoOp!^draft_d1/StatefulPartitionedCall!^draft_d3/StatefulPartitionedCall-^draft_reconstruction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2D
 draft_d1/StatefulPartitionedCall draft_d1/StatefulPartitionedCall2D
 draft_d3/StatefulPartitionedCall draft_d3/StatefulPartitionedCall2\
,draft_reconstruction/StatefulPartitionedCall,draft_reconstruction/StatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedraft_d1_input
�
�
!__inference__traced_restore_54289
file_prefix4
 assignvariableop_draft_d1_kernel:
��/
 assignvariableop_1_draft_d1_bias:	�6
"assignvariableop_2_draft_d3_kernel:
��/
 assignvariableop_3_draft_d3_bias:	�C
.assignvariableop_4_draft_reconstruction_kernel:���<
,assignvariableop_5_draft_reconstruction_bias:
��

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_draft_d1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_draft_d1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_draft_d3_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_draft_d3_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp.assignvariableop_4_draft_reconstruction_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp,assignvariableop_5_draft_reconstruction_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
C__inference_draft_d3_layer_call_and_return_conditional_losses_53859

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_draft_d1_layer_call_fn_54151

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_draft_d1_layer_call_and_return_conditional_losses_53842p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_draft_d1_layer_call_and_return_conditional_losses_53842

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_sequential_2_layer_call_and_return_conditional_losses_54117

inputs;
'draft_d1_matmul_readvariableop_resource:
��7
(draft_d1_biasadd_readvariableop_resource:	�;
'draft_d3_matmul_readvariableop_resource:
��7
(draft_d3_biasadd_readvariableop_resource:	�H
3draft_reconstruction_matmul_readvariableop_resource:���D
4draft_reconstruction_biasadd_readvariableop_resource:
��
identity��draft_d1/BiasAdd/ReadVariableOp�draft_d1/MatMul/ReadVariableOp�draft_d3/BiasAdd/ReadVariableOp�draft_d3/MatMul/ReadVariableOp�+draft_reconstruction/BiasAdd/ReadVariableOp�*draft_reconstruction/MatMul/ReadVariableOp�
draft_d1/MatMul/ReadVariableOpReadVariableOp'draft_d1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0|
draft_d1/MatMulMatMulinputs&draft_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
draft_d1/BiasAdd/ReadVariableOpReadVariableOp(draft_d1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
draft_d1/BiasAddBiasAdddraft_d1/MatMul:product:0'draft_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
draft_d1/ReluReludraft_d1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
draft_d3/MatMul/ReadVariableOpReadVariableOp'draft_d3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
draft_d3/MatMulMatMuldraft_d1/Relu:activations:0&draft_d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
draft_d3/BiasAdd/ReadVariableOpReadVariableOp(draft_d3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
draft_d3/BiasAddBiasAdddraft_d3/MatMul:product:0'draft_d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
draft_d3/ReluReludraft_d3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
*draft_reconstruction/MatMul/ReadVariableOpReadVariableOp3draft_reconstruction_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
draft_reconstruction/MatMulMatMuldraft_d3/Relu:activations:02draft_reconstruction/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
+draft_reconstruction/BiasAdd/ReadVariableOpReadVariableOp4draft_reconstruction_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
draft_reconstruction/BiasAddBiasAdd%draft_reconstruction/MatMul:product:03draft_reconstruction/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
draft_reconstruction/SigmoidSigmoid%draft_reconstruction/BiasAdd:output:0*
T0*)
_output_shapes
:�����������q
IdentityIdentity draft_reconstruction/Sigmoid:y:0^NoOp*
T0*)
_output_shapes
:������������
NoOpNoOp ^draft_d1/BiasAdd/ReadVariableOp^draft_d1/MatMul/ReadVariableOp ^draft_d3/BiasAdd/ReadVariableOp^draft_d3/MatMul/ReadVariableOp,^draft_reconstruction/BiasAdd/ReadVariableOp+^draft_reconstruction/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2B
draft_d1/BiasAdd/ReadVariableOpdraft_d1/BiasAdd/ReadVariableOp2@
draft_d1/MatMul/ReadVariableOpdraft_d1/MatMul/ReadVariableOp2B
draft_d3/BiasAdd/ReadVariableOpdraft_d3/BiasAdd/ReadVariableOp2@
draft_d3/MatMul/ReadVariableOpdraft_d3/MatMul/ReadVariableOp2Z
+draft_reconstruction/BiasAdd/ReadVariableOp+draft_reconstruction/BiasAdd/ReadVariableOp2X
*draft_reconstruction/MatMul/ReadVariableOp*draft_reconstruction/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_sequential_2_layer_call_and_return_conditional_losses_54142

inputs;
'draft_d1_matmul_readvariableop_resource:
��7
(draft_d1_biasadd_readvariableop_resource:	�;
'draft_d3_matmul_readvariableop_resource:
��7
(draft_d3_biasadd_readvariableop_resource:	�H
3draft_reconstruction_matmul_readvariableop_resource:���D
4draft_reconstruction_biasadd_readvariableop_resource:
��
identity��draft_d1/BiasAdd/ReadVariableOp�draft_d1/MatMul/ReadVariableOp�draft_d3/BiasAdd/ReadVariableOp�draft_d3/MatMul/ReadVariableOp�+draft_reconstruction/BiasAdd/ReadVariableOp�*draft_reconstruction/MatMul/ReadVariableOp�
draft_d1/MatMul/ReadVariableOpReadVariableOp'draft_d1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0|
draft_d1/MatMulMatMulinputs&draft_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
draft_d1/BiasAdd/ReadVariableOpReadVariableOp(draft_d1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
draft_d1/BiasAddBiasAdddraft_d1/MatMul:product:0'draft_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
draft_d1/ReluReludraft_d1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
draft_d3/MatMul/ReadVariableOpReadVariableOp'draft_d3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
draft_d3/MatMulMatMuldraft_d1/Relu:activations:0&draft_d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
draft_d3/BiasAdd/ReadVariableOpReadVariableOp(draft_d3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
draft_d3/BiasAddBiasAdddraft_d3/MatMul:product:0'draft_d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
draft_d3/ReluReludraft_d3/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
*draft_reconstruction/MatMul/ReadVariableOpReadVariableOp3draft_reconstruction_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
draft_reconstruction/MatMulMatMuldraft_d3/Relu:activations:02draft_reconstruction/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
+draft_reconstruction/BiasAdd/ReadVariableOpReadVariableOp4draft_reconstruction_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
draft_reconstruction/BiasAddBiasAdd%draft_reconstruction/MatMul:product:03draft_reconstruction/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
draft_reconstruction/SigmoidSigmoid%draft_reconstruction/BiasAdd:output:0*
T0*)
_output_shapes
:�����������q
IdentityIdentity draft_reconstruction/Sigmoid:y:0^NoOp*
T0*)
_output_shapes
:������������
NoOpNoOp ^draft_d1/BiasAdd/ReadVariableOp^draft_d1/MatMul/ReadVariableOp ^draft_d3/BiasAdd/ReadVariableOp^draft_d3/MatMul/ReadVariableOp,^draft_reconstruction/BiasAdd/ReadVariableOp+^draft_reconstruction/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2B
draft_d1/BiasAdd/ReadVariableOpdraft_d1/BiasAdd/ReadVariableOp2@
draft_d1/MatMul/ReadVariableOpdraft_d1/MatMul/ReadVariableOp2B
draft_d3/BiasAdd/ReadVariableOpdraft_d3/BiasAdd/ReadVariableOp2@
draft_d3/MatMul/ReadVariableOpdraft_d3/MatMul/ReadVariableOp2Z
+draft_reconstruction/BiasAdd/ReadVariableOp+draft_reconstruction/BiasAdd/ReadVariableOp2X
*draft_reconstruction/MatMul/ReadVariableOp*draft_reconstruction/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
,__inference_sequential_2_layer_call_fn_54092

inputs
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:���
	unknown_4:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_53960q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_draft_d3_layer_call_fn_54171

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_draft_d3_layer_call_and_return_conditional_losses_53859p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_sequential_2_layer_call_and_return_conditional_losses_53960

inputs"
draft_d1_53944:
��
draft_d1_53946:	�"
draft_d3_53949:
��
draft_d3_53951:	�/
draft_reconstruction_53954:���*
draft_reconstruction_53956:
��
identity�� draft_d1/StatefulPartitionedCall� draft_d3/StatefulPartitionedCall�,draft_reconstruction/StatefulPartitionedCall�
 draft_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdraft_d1_53944draft_d1_53946*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_draft_d1_layer_call_and_return_conditional_losses_53842�
 draft_d3/StatefulPartitionedCallStatefulPartitionedCall)draft_d1/StatefulPartitionedCall:output:0draft_d3_53949draft_d3_53951*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_draft_d3_layer_call_and_return_conditional_losses_53859�
,draft_reconstruction/StatefulPartitionedCallStatefulPartitionedCall)draft_d3/StatefulPartitionedCall:output:0draft_reconstruction_53954draft_reconstruction_53956*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_draft_reconstruction_layer_call_and_return_conditional_losses_53876�
IdentityIdentity5draft_reconstruction/StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:������������
NoOpNoOp!^draft_d1/StatefulPartitionedCall!^draft_d3/StatefulPartitionedCall-^draft_reconstruction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2D
 draft_d1/StatefulPartitionedCall draft_d1/StatefulPartitionedCall2D
 draft_d3/StatefulPartitionedCall draft_d3/StatefulPartitionedCall2\
,draft_reconstruction/StatefulPartitionedCall,draft_reconstruction/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_sequential_2_layer_call_and_return_conditional_losses_53924

inputs"
draft_d1_53908:
��
draft_d1_53910:	�"
draft_d3_53913:
��
draft_d3_53915:	�/
draft_reconstruction_53918:���*
draft_reconstruction_53920:
��
identity�� draft_d1/StatefulPartitionedCall� draft_d3/StatefulPartitionedCall�,draft_reconstruction/StatefulPartitionedCall�
 draft_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdraft_d1_53908draft_d1_53910*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_draft_d1_layer_call_and_return_conditional_losses_53842�
 draft_d3/StatefulPartitionedCallStatefulPartitionedCall)draft_d1/StatefulPartitionedCall:output:0draft_d3_53913draft_d3_53915*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_draft_d3_layer_call_and_return_conditional_losses_53859�
,draft_reconstruction/StatefulPartitionedCallStatefulPartitionedCall)draft_d3/StatefulPartitionedCall:output:0draft_reconstruction_53918draft_reconstruction_53920*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_draft_reconstruction_layer_call_and_return_conditional_losses_53876�
IdentityIdentity5draft_reconstruction/StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:������������
NoOpNoOp!^draft_d1/StatefulPartitionedCall!^draft_d3/StatefulPartitionedCall-^draft_reconstruction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2D
 draft_d1/StatefulPartitionedCall draft_d1/StatefulPartitionedCall2D
 draft_d3/StatefulPartitionedCall draft_d3/StatefulPartitionedCall2\
,draft_reconstruction/StatefulPartitionedCall,draft_reconstruction/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_sequential_2_layer_call_and_return_conditional_losses_53883
draft_d1_input"
draft_d1_53843:
��
draft_d1_53845:	�"
draft_d3_53860:
��
draft_d3_53862:	�/
draft_reconstruction_53877:���*
draft_reconstruction_53879:
��
identity�� draft_d1/StatefulPartitionedCall� draft_d3/StatefulPartitionedCall�,draft_reconstruction/StatefulPartitionedCall�
 draft_d1/StatefulPartitionedCallStatefulPartitionedCalldraft_d1_inputdraft_d1_53843draft_d1_53845*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_draft_d1_layer_call_and_return_conditional_losses_53842�
 draft_d3/StatefulPartitionedCallStatefulPartitionedCall)draft_d1/StatefulPartitionedCall:output:0draft_d3_53860draft_d3_53862*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_draft_d3_layer_call_and_return_conditional_losses_53859�
,draft_reconstruction/StatefulPartitionedCallStatefulPartitionedCall)draft_d3/StatefulPartitionedCall:output:0draft_reconstruction_53877draft_reconstruction_53879*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_draft_reconstruction_layer_call_and_return_conditional_losses_53876�
IdentityIdentity5draft_reconstruction/StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:������������
NoOpNoOp!^draft_d1/StatefulPartitionedCall!^draft_d3/StatefulPartitionedCall-^draft_reconstruction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2D
 draft_d1/StatefulPartitionedCall draft_d1/StatefulPartitionedCall2D
 draft_d3/StatefulPartitionedCall draft_d3/StatefulPartitionedCall2\
,draft_reconstruction/StatefulPartitionedCall,draft_reconstruction/StatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedraft_d1_input
�:
�
__inference__traced_save_54261
file_prefix:
&read_disablecopyonread_draft_d1_kernel:
��5
&read_1_disablecopyonread_draft_d1_bias:	�<
(read_2_disablecopyonread_draft_d3_kernel:
��5
&read_3_disablecopyonread_draft_d3_bias:	�I
4read_4_disablecopyonread_draft_reconstruction_kernel:���B
2read_5_disablecopyonread_draft_reconstruction_bias:
��
savev2_const
identity_13��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_draft_d1_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_draft_d1_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_draft_d1_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_draft_d1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_draft_d3_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_draft_d3_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_draft_d3_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_draft_d3_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_4/DisableCopyOnReadDisableCopyOnRead4read_4_disablecopyonread_draft_reconstruction_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp4read_4_disablecopyonread_draft_reconstruction_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:���*
dtype0p

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:���f

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*!
_output_shapes
:����
Read_5/DisableCopyOnReadDisableCopyOnRead2read_5_disablecopyonread_draft_reconstruction_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp2read_5_disablecopyonread_draft_reconstruction_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:��*
dtype0l
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:��c
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:���
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
	2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_12Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_13IdentityIdentity_12:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*#
_input_shapes
: : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
,__inference_sequential_2_layer_call_fn_53975
draft_d1_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:���
	unknown_4:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldraft_d1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_53960q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedraft_d1_input
�

�
O__inference_draft_reconstruction_layer_call_and_return_conditional_losses_54202

inputs3
matmul_readvariableop_resource:���/
biasadd_readvariableop_resource:
��
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0k
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������t
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:��*
dtype0x
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������X
SigmoidSigmoidBiasAdd:output:0*
T0*)
_output_shapes
:�����������\
IdentityIdentitySigmoid:y:0^NoOp*
T0*)
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
4__inference_draft_reconstruction_layer_call_fn_54191

inputs
unknown:���
	unknown_0:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_draft_reconstruction_layer_call_and_return_conditional_losses_53876q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
,__inference_sequential_2_layer_call_fn_53939
draft_d1_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:���
	unknown_4:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldraft_d1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_53924q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedraft_d1_input
�	
�
,__inference_sequential_2_layer_call_fn_54075

inputs
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:���
	unknown_4:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_53924q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_draft_d1_layer_call_and_return_conditional_losses_54162

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_54058
draft_d1_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:���
	unknown_4:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldraft_d1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_53827q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedraft_d1_input
�

�
C__inference_draft_d3_layer_call_and_return_conditional_losses_54182

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
J
draft_d1_input8
 serving_default_draft_d1_input:0����������J
draft_reconstruction2
StatefulPartitionedCall:0�����������tensorflow/serving/predict:�e
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
J
0
1
2
3
"4
#5"
trackable_list_wrapper
J
0
1
2
3
"4
#5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
�
)trace_0
*trace_1
+trace_2
,trace_32�
,__inference_sequential_2_layer_call_fn_53939
,__inference_sequential_2_layer_call_fn_53975
,__inference_sequential_2_layer_call_fn_54075
,__inference_sequential_2_layer_call_fn_54092�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z)trace_0z*trace_1z+trace_2z,trace_3
�
-trace_0
.trace_1
/trace_2
0trace_32�
G__inference_sequential_2_layer_call_and_return_conditional_losses_53883
G__inference_sequential_2_layer_call_and_return_conditional_losses_53902
G__inference_sequential_2_layer_call_and_return_conditional_losses_54117
G__inference_sequential_2_layer_call_and_return_conditional_losses_54142�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z-trace_0z.trace_1z/trace_2z0trace_3
�B�
 __inference__wrapped_model_53827draft_d1_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
1serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
7trace_02�
(__inference_draft_d1_layer_call_fn_54151�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z7trace_0
�
8trace_02�
C__inference_draft_d1_layer_call_and_return_conditional_losses_54162�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z8trace_0
#:!
��2draft_d1/kernel
:�2draft_d1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
>trace_02�
(__inference_draft_d3_layer_call_fn_54171�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>trace_0
�
?trace_02�
C__inference_draft_d3_layer_call_and_return_conditional_losses_54182�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z?trace_0
#:!
��2draft_d3/kernel
:�2draft_d3/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
Etrace_02�
4__inference_draft_reconstruction_layer_call_fn_54191�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zEtrace_0
�
Ftrace_02�
O__inference_draft_reconstruction_layer_call_and_return_conditional_losses_54202�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zFtrace_0
0:.���2draft_reconstruction/kernel
):'��2draft_reconstruction/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_sequential_2_layer_call_fn_53939draft_d1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_2_layer_call_fn_53975draft_d1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_2_layer_call_fn_54075inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_sequential_2_layer_call_fn_54092inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_2_layer_call_and_return_conditional_losses_53883draft_d1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_2_layer_call_and_return_conditional_losses_53902draft_d1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_2_layer_call_and_return_conditional_losses_54117inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_sequential_2_layer_call_and_return_conditional_losses_54142inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_54058draft_d1_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_draft_d1_layer_call_fn_54151inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_draft_d1_layer_call_and_return_conditional_losses_54162inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_draft_d3_layer_call_fn_54171inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_draft_d3_layer_call_and_return_conditional_losses_54182inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_draft_reconstruction_layer_call_fn_54191inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_draft_reconstruction_layer_call_and_return_conditional_losses_54202inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_53827�"#8�5
.�+
)�&
draft_d1_input����������
� "M�J
H
draft_reconstruction0�-
draft_reconstruction������������
C__inference_draft_d1_layer_call_and_return_conditional_losses_54162e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_draft_d1_layer_call_fn_54151Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_draft_d3_layer_call_and_return_conditional_losses_54182e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_draft_d3_layer_call_fn_54171Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
O__inference_draft_reconstruction_layer_call_and_return_conditional_losses_54202f"#0�-
&�#
!�
inputs����������
� ".�+
$�!
tensor_0�����������
� �
4__inference_draft_reconstruction_layer_call_fn_54191["#0�-
&�#
!�
inputs����������
� "#� 
unknown������������
G__inference_sequential_2_layer_call_and_return_conditional_losses_53883z"#@�=
6�3
)�&
draft_d1_input����������
p

 
� ".�+
$�!
tensor_0�����������
� �
G__inference_sequential_2_layer_call_and_return_conditional_losses_53902z"#@�=
6�3
)�&
draft_d1_input����������
p 

 
� ".�+
$�!
tensor_0�����������
� �
G__inference_sequential_2_layer_call_and_return_conditional_losses_54117r"#8�5
.�+
!�
inputs����������
p

 
� ".�+
$�!
tensor_0�����������
� �
G__inference_sequential_2_layer_call_and_return_conditional_losses_54142r"#8�5
.�+
!�
inputs����������
p 

 
� ".�+
$�!
tensor_0�����������
� �
,__inference_sequential_2_layer_call_fn_53939o"#@�=
6�3
)�&
draft_d1_input����������
p

 
� "#� 
unknown������������
,__inference_sequential_2_layer_call_fn_53975o"#@�=
6�3
)�&
draft_d1_input����������
p 

 
� "#� 
unknown������������
,__inference_sequential_2_layer_call_fn_54075g"#8�5
.�+
!�
inputs����������
p

 
� "#� 
unknown������������
,__inference_sequential_2_layer_call_fn_54092g"#8�5
.�+
!�
inputs����������
p 

 
� "#� 
unknown������������
#__inference_signature_wrapper_54058�"#J�G
� 
@�=
;
draft_d1_input)�&
draft_d1_input����������"M�J
H
draft_reconstruction0�-
draft_reconstruction�����������