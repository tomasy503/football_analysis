from utils import read_video, save_video


def main():
    # Read video
    video_frames = read_video("data/raw/bmg_test.mp4")

    # Save video
    save_video(video_frames, "data/processed/output_video.mp4")


if __name__ == "__main__":
    main()
