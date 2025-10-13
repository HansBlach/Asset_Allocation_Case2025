def returns_to_value(data, start_value = 1):
    value_data = np.zeros(data.shape)
    value_data[0,:] = start_value

    for i in range(1, data.shape[0]):
        value_data[i, :] = value_data[i-1, :] * (1+data[i, :])
    
    return value_data