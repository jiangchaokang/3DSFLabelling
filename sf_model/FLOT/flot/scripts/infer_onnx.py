import onnxruntime, onnx, torch
import time

onnx_path = './SF_Model/TOOL/model/scene_flow_2048_nb8.onnx' 

cuda_id =0
pc1 = torch.randn(1, 2048, 3)
pc2 = torch.randn(1, 2048, 3)

providers = [
    ('CUDAExecutionProvider', {
        'device_id': cuda_id,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]   
onnx_session = onnxruntime.InferenceSession(
    onnx_path,
    providers = providers
)

print(onnx_session.get_providers())

input_name1 = onnx_session.get_inputs()[0].name
input_name2 = onnx_session.get_inputs()[1].name
output_name = onnx_session.get_outputs()[0].name
pc1 = pc1.numpy()
pc2 = pc2.numpy()
start_time = time.time()
left_onnx_result = onnx_session.run([output_name], {input_name1: pc1, input_name2: pc2})[0].squeeze()
print(left_onnx_result)
run_times = time.time() - start_time
print("run_times:", run_times)

# import pdb
# pdb.set_trace()