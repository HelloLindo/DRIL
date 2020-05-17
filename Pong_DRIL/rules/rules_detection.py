'''
Rules Detection Module
[Safety Rules]
Rule 1: Move the paddle only if the ball is moving on the (Dynamically) RIL agent’s side.
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
    global frame_bgr
    # Ball part index
    frame_bgr_cut = frame_bgr[13:78, 40:73]
    frame_cut = np.float32(frame[13:78, 40:73])
    corner_set = cv2.cornerHarris(frame_cut, 2, 3, 0.04)

    (x, y), radius = cv2.minEnclosingCircle(np.argwhere(corner_set > 0.01 * corner_set.max()))
    (x, y) = np.int0((x, y))

    ''' Show center index of frame array '''
    # print("The center index of this frame's corners is " + (x, y))
    ''' Draw and show figure '''
    frame_bgr_cut[corner_set > 0.01 * corner_set.max()] = [0, 255, 0]
    frame_bgr_cut[x, y] = [0, 0, 255]
    # cv_show(frame_bgr_cut)

    return x, y

'''
    Input: full game cv frame array
    Output: right board center index in cv frame array
'''
def extract_right_board(frame):
    global frame_bgr
    # Right board part index
    frame_bgr_cut = frame_bgr[13:78, 72:80]
    frame_cut = frame[13:78, 72:80]
    corner_set = cv2.cornerHarris(frame_cut, 2, 3, 0.04)

    (x, y), radius = cv2.minEnclosingCircle(np.argwhere(corner_set > 0.01 * corner_set.max()))
    (x, y) = np.int0((x, y))

    ''' Show center index of frame array '''
    # print("The center index of this frame's corners is " + (x, y))
    ''' Draw and show figure '''
    frame_bgr_cut[corner_set > 0.01 * corner_set.max()] = [255, 0, 0]
    frame_bgr_cut[x-3:x+3, y] = [0, 0, 255]
    # cv_show(frame_bgr_cut)

    return x, y


def state_tensor_to_cv(state_tensor):
    '''
        Convert State Tensor to cv2 frame.
    '''
    global frame_bgr
    state_tensor = np.array(state_tensor)
    state_tensor = state_tensor.transpose((1, 2, 0))

    frame = cv2.cvtColor(state_tensor, cv2.COLOR_RGBA2GRAY)

    # Codes below only for test
    frame_bgr = cv2.cvtColor(state_tensor, cv2.COLOR_RGBA2BGR)
    # cv_show(frame)
    # cv_show(frame_bgr)

    return frame

def choose_action_by_rules(state_tensor):
    state_tensor = state_tensor.squeeze(0)
    fetch_state = state_tensor[3, :, :]
    fetch_state = fetch_state.unsqueeze(0)
    fetch_state = fetch_state.repeat([4, 1, 1])

    # frame = state_tensor_to_cv(state_tensor)
    frame = state_tensor_to_cv(fetch_state)

    ball_x, ball_y = extract_ball(frame)
    right_board_x, right_board_y = extract_right_board(frame)

    # cv_show(frame_bgr, True)

    if ball_x == 0:
        ''' No detect ball return Freeze '''
        return torch.tensor([[1]])

    # print(right_board_x, ball_x, right_board_x < ball_x)

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