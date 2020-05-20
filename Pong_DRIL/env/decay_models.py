def get_exponential_decay(params, episode_counter):
    ''' f(x) =  EXP_INITIAL_OMEGA * ( EXP_DECAY_RATE^(x/EXP_DECAY_STEPS) )'''
    omega = params["EXP_INITIAL_OMEGA"] * (params["EXP_DECAY_RATE"] ** (episode_counter / params["EXP_DECAY_STEPS"]))
    # return omega if omega > 0.1 else 0.1
    return omega

def get_quad_decay(params, episode_counter):
    ''' f(x) = QUAD_DECAY_XX * x^2 + QUAD_DECAY_X * x + QUAD_DECAY_CONST '''
    omega = params["QUAD_DECAY_XX"] * (episode_counter ** 2) + params["QUAD_DECAY_X"] * episode_counter + params["QUAD_DECAY_CONST"]
    return omega if omega > 0.1 else 0.1

def get_polynomial_decay(params, episode_counter):
    ''' f(x) = QUAD_DECAY_XXX * x^3 + QUAD_DECAY_XX * x^2 + QUAD_DECAY_X * x + QUAD_DECAY_CONST '''
    omega = params["POLY_DECAY_XXX"] * (episode_counter ** 3) + params["POLY_DECAY_XX"] * (episode_counter ** 2) + params["POLY_DECAY_X"] * episode_counter + params["POLY_DECAY_CONST"]
    return omega if omega > 0.1 else 0.1

def get_linear_decay(params, episode_counter):
    ''' f(x) = LINEAR_DECAY_X * x + LINEAR_DECAY_CONST '''
    omega = params["LINEAR_DECAY_X"] * episode_counter + params["LINEAR_DECAY_CONST"]
    return omega if omega > 0.1 else 0.1