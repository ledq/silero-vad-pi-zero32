import onnxruntime
import numpy as np
import pyaudio
import queue

session = onnxruntime.InferenceSession("silero_vad.onnx")


input_names = [i.name for i in session.get_inputs()]
print(f"Detected input names: {input_names}")

input_name_audio = input_names[0]  # 'input'
input_name_state = input_names[1]  # 'state'
input_name_sr = input_names[2]     # 'sr'

# detect type for state
state_dtype = np.int64 if session.get_inputs()[1].type == 'tensor(int64)' else np.float32
sr_dtype = np.int64 if session.get_inputs()[2].type == 'tensor(int64)' else np.float32

# initialize state and sampling rate
state = np.zeros((2, 1, 128), dtype=state_dtype)  # packed hidden and cell
sr = np.array([16000], dtype=sr_dtype)


sample_rate = 16000
frames_per_buffer = 512
q = queue.Queue()

def callback(in_data, frame_count, time_info, status):
    q.put(in_data)
    return (None, pyaudio.paContinue)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=0,
                frames_per_buffer=frames_per_buffer,
                stream_callback=callback)
stream.start_stream()

print("Listening...")

try:
    while True:
        if not q.empty():
            raw_audio = q.get()
            samples = np.frombuffer(raw_audio, dtype=np.int16)
            #print(f"First samples: {samples[:10]}")
            # convert raw audio to float32 in [-1.0, 1.0]
            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            #print(f"Mean amplitude after scaling: {np.mean(np.abs(audio_np))}")
            audio_np = np.expand_dims(audio_np, axis=0)  # (1, N)

            # run inference
            outputs = session.run(None, {
                input_name_audio: audio_np,
                input_name_state: state,
                input_name_sr: sr
            })

            # outputs reading
            speech_prob = outputs[0][0][0]
            state = outputs[1]   # updated packed hidden+cell state

            print(f"Speech probability: {speech_prob:.3f}")

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()
