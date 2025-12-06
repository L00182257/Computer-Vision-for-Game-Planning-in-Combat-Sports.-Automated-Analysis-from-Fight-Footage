import cv2
from video_io import iterate_frames

VIDEO_PATH = "C:/Users/noahr/Downloads - Copy/Downloads/Computer Vision for Game-Planning in Combat Sports. Automated Analysis from imFight Footage/data/raw/MH vs JG RND4 Clip1.mp4"

def main():
    for idx, ts, frame in iterate_frames(VIDEO_PATH, stride=10):
        print(f"Frame {idx} @ {ts:.2f}s")

        # Show frame for debugging
        cv2.imshow("Test Frame", frame)

        # Exit if you press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()