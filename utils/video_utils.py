import cv2


def read_video(video_path):
    capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(ouput_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24.0,
        (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]),
    )
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()
