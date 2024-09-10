from trackers import Tracker
from utils import read_video, save_video


def main():
    # Read video
    video_frames = read_video("data/raw/bmg_test.mp4")

    # Initialize tracker
    tracker = Tracker("model/best.pt")

    # Detect objects in video frames
    detections = tracker.object_detection(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )

    # draw output
    # draw object annotations
    output_video_frames = tracker.draw_annoatations(video_frames, detections)

    # Save video
    save_video(output_video_frames, "data/processed/output_video.mp4")


if __name__ == "__main__":
    main()
