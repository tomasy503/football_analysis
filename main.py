import numpy as np

from player_ball_assignment import PlayerBallAssigner
from team_assignment import TeamAssignment
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

    # Interpolate ball positions
    detections["ball"] = tracker.interpolate_ball_position(detections["ball"])

    # Assign teams
    team_assigner = TeamAssignment()
    team_assigner.assign_team_color(video_frames[0], detections["players"][0])

    for frame_num, player_detection in enumerate(detections["players"]):
        for player_id, detection in player_detection.items():
            team = team_assigner.assign_team(
                video_frames[frame_num], detection["bbox"], player_id
            )
            detections["players"][frame_num][player_id]["team"] = team
            detections["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )
    # Assign ball to player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_detection in enumerate(detections["players"]):
        ball_bbox = detections["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(
            player_detection, ball_bbox
        )

        if assigned_player != -1:
            detections["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                detections["players"][frame_num][assigned_player]["team"]
            )
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    ##### draw output
    # draw object annotations
    output_video_frames = tracker.draw_annotations(
        video_frames, detections, team_ball_control
    )

    # Save video
    save_video(output_video_frames, "data/processed/output_video.mp4")


if __name__ == "__main__":
    main()
