import torch
import tltorch

def get_random_direction_TT(tt_parameter):
    '''
    given a weight in TT format, 
    return a random direction for the parameter in TT format
    '''
    # get random_direction_matrix with the shape of (in_features, out_features)
    random_direction_matrix = torch.rand((tt_parameter.in_features, tt_parameter.out_features), 
                                         device=tt_parameter.device, dtype=tt_parameter.dtype)
    # for each neuron, normalize it using the norm of the reconstructed weight matrix
    weight_matrix = tt_parameter.tensor.to_tensor().reshape(tt_parameter.in_features, 
                                                            tt_parameter.out_features)
    for i in range(tt_parameter.out_features):
        normalized = random_direction_matrix[:,i] / random_direction_matrix[:,i].norm() \
            * weight_matrix[:,i].norm()
        random_direction_matrix[:,i] = normalized
    # and tensorize it and decompose into TT format
    random_direction_tensor = random_direction_matrix.reshape(tt_parameter.tensorized_dim)
    print(tt_parameter.tensor.rank)
    tt_direction = tltorch.TTTensor.new(tt_parameter.tensorized_dim, rank=tt_parameter.tensor.rank)
    
    return tt_direction

def get_random_direction_mat(parameter):
    '''
    given a weight in matrix format,
    return a random direction for the parameter in matrix format
    '''

    random_direction = torch.rand_like(parameter)
    for i in range(parameter.shape[0]):
        random_direction[i,:] = random_direction[i,:] / random_direction[:,i].norm() \
            * parameter[i,:].norm()
    
    return random_direction

def get_moved_parameter_tt(tt_parameter, tt_direction, alpha):
    for p_factor, d_factor in zip(tt_parameter.tensor.factors, tt_direction.factors):
        p_factor += d_factor * alpha
    return tt_parameter

def visualize_loss(model, state_dict, loss_function):

    model.load_state_dict(state_dict)

    random_direction_1 = get_random_direction_TT(model.layer_1.weight_tensor)
    random_direction_2 = get_random_direction_TT(model.layer_2.weight_tensor)
    random_direction_3 = get_random_direction_mat(model.layer_3.weight)
    
    losses = []
    for i in range(-1,1.01,0.01):
        model.load_state_dict(state_dict)
        
    pass