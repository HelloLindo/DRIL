'''
Rules Detection Module
[Acceleration Rules]
Rule 1: When the ball is in the upper of the agent’s paddle -> move the paddle up
Rule 2: When the ball is in the lower of the agent’s paddle -> move the paddle down
'''
import torch
import numpy as np
import cv2

frame_bgr = None


def cv_show(cv_img, zoom_up=False):
    if zoom_up:
        cv2.namedWindow("Extraction of Pong", 0);
        cv2.resizeWindow("Extraction of Pong", cv_img.shape[1] * 5, cv_img.shape[0] * 5);
    cv2.imshow("Extraction of Pong", cv_img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

'''
    Input: full game cv frame array
    Output: ball center index in cv frame array
'''
def extract_ball(frame):
    # Ball part index
    ''' No Safety: Use the full side of the game window, not the agent's side. '''
    frame_cut = frame[13:78, 11:73]
    corner_set = cv2.cornerHarris(frame_cut, 2, 3, 0.04)

    (x, y), radius = cv2.minEnclosingCircle(np.argwhere(corner_set > 0.01 * corner_set.max()))
    (x, y) = np.int0((x, y))

    ''' Show center index of frame array - Only for Debug '''
    # print("The center index of this frame's corners is " + (x, y))
    return x, y

'''
    Input: full game cv frame array
    Output: right board center index in cv frame array
'''
def extract_right_board(frame):
    # Right board part index
    frame_cut = frame[13:78, 72:80]
    corner_set = cv2.cornerHarris(frame_cut, 2, 3, 0.04)

    (x, y), radius = cv2.minEnclosingCircle(np.argwhere(corner_set > 0.01 * corner_set.max()))
    (x, y) = np.int0((x, y))

    ''' Show center index of frame array - Only for Debug '''
    # print("The center index of this frame's corners is " + (x, y))
    return x, y

def choose_action_by_rules(state_tensor):
    state_tensor = state_tensor.squeeze(0)
    fetch_state = state_tensor[3, :, :]
    frame = fetch_state.numpy()

    ball_x, ball_y = extract_ball(frame)
    right_board_x, right_board_y = extract_right_board(frame)

    ''' Show the extraction window - Only for Debug'''
    # cv_show(frame, True)

    if ball_x == 0:
        ''' ball detection failed -> return Freeze '''
        return torch.tensor([[1]])

    if (-5 < right_board_x - ball_x < 0) or (0 < right_board_x - ball_x < 5):
        ''' Freeze '''
        return torch.tensor([[1]])
    elif right_board_x-3 < ball_x:
        ''' Move Down '''
        return torch.tensor([[3]])
    elif right_board_x+3 > ball_x:
        ''' Move Up '''
        return torch.tensor([[2]])
    else:
        ''' Freeze '''
        return torch.tensor([[1]])