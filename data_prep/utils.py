import numpy as np
import matplotlib.pyplot as plt

def Green_func(x,y):
    return 1/2 * (x + y - np.abs(y-x)) - x*y

def model_prediction(model, input_signals):
    if input_signals.ndim != 3:
        input_signals = input_signals[:,np.newaxis,:]

    predictions = np.zeros_like(input_signals)
    for i in range(input_signals.shape[0]):
        predictions[i,:,:] = model(input_signals[i,:,:])

    return predictions

def load_friction_data():
    feature = np.genfromtxt('friction_data/features_SlipLaw_v2.csv', delimiter=',')[:, np.newaxis,:]
    target = np.genfromtxt('friction_data/targets_SlipLaw_v2.csv', delimiter=',')[:, np.newaxis,:]

    train_x = feature[:750,:,:]
    train_y = target[:750,:,:]
    test_x = feature[750:,:,:]
    test_y = target[750:,:,:]

    x_max = train_x.max()

    train_x_norm = np.log(np.log(1/train_x))
    test_x_norm = np.log(np.log(1/test_x))
    y_max = train_y.max()

    train_y_norm = train_y
    test_y_norm = test_y 

    return train_x_norm, train_y_norm, test_x_norm, test_y_norm

def impulse_response(model, N=300):
    impulse = np.zeros(N)
    impulse[0] = 1
    impulse = impulse.reshape(1,-1)[:,np.newaxis,:]
    pred = model(impulse[0,:,:])[0]
    return pred

def generate_random_fourierSignals(time, samples, num_of_coeffs = 3,
                                   freq_range=(0, 0.1), multiplier_range=(-2, 2)):
    
    phase = np.random.uniform(0, 2*np.pi, (samples, num_of_coeffs))
    freqs = np.random.uniform(freq_range[0], freq_range[1], (samples, num_of_coeffs))
    multipliers = np.random.uniform(multiplier_range[0], multiplier_range[1], (samples, num_of_coeffs))

    input_signals = np.zeros((samples, len(time)))

    for j in range(samples):
        for i in range(num_of_coeffs):
            input_signals[j] += multipliers[j, i] * np.sin(2 * np.pi * freqs[j, i] * time + phase[j, i])

    return input_signals

def generate_random_pulse(time, samples):

    N = time.shape[0]
    input_signals = np.zeros((samples, len(time)))

    start_impulse = np.random.randint(20, N-20, size = samples)
    for i in range(samples):
        max_time_for_impulse = N - start_impulse[i] - 20
        length_impulse = np.random.randint(5, max_time_for_impulse)
        input_signals[i, start_impulse[i]:length_impulse + start_impulse[i]] = 1

    return input_signals


def generate_pulse_response(input_signal_type, time, samples=100, H_hat=None):
    
    if H_hat is None:
        H_hat = np.load("pulse_data/H_hat.npy")

    len_H_hat = H_hat.shape[0]
    len_time = len(time)

    # Create a dummy time vector at the H_hat resolution for input generation
    time_H_hat = np.linspace(0, 200, len_H_hat)

    # Generate input signals at H_hat resolution
    if input_signal_type == 'fourier':
        input_signals_highres = generate_random_fourierSignals(time_H_hat, samples)
    elif input_signal_type == 'pulse':
        input_signals_highres = generate_random_pulse(time_H_hat, samples)
    else:
        raise ValueError("Unknown input_signal_type")

    output_signals_highres = np.zeros_like(input_signals_highres)

    # Perform convolution in frequency domain
    for i, sig in enumerate(input_signals_highres):
        u = np.fft.ifft(H_hat * np.fft.fft(sig))
        output_signals_highres[i] = u.real  # Only real part is physical

    # Downsample input and output signals to match `time` length
    input_signals = input_signals_highres[:, ::len_H_hat // len_time]
    output_signals = output_signals_highres[:, ::len_H_hat // len_time]

    # Ensure exact final shape by trimming if oversampled
    input_signals = input_signals[:, :len_time]
    output_signals = output_signals[:, :len_time]

    return input_signals, output_signals

def generate_green_response(time, N = 50, samples=2000, num_of_coeffs=10,
                        freq_range=(0, 0.1), multiplier_range=(-2, 2)):


    green_over_midpoint_grid = Green_func(time[:, None], time[None, :])

    input_signals = generate_random_fourierSignals(time, samples, num_of_coeffs,
                                                   freq_range, multiplier_range)

    output_signals = np.zeros_like(input_signals)

    for i,sig in enumerate(input_signals):
        u = np.trapz(green_over_midpoint_grid * sig, time, axis=1)
        output_signals[i] = u

    return input_signals, output_signals


def plot_pred_v_truth(x, pred, truth):
    # Determine number of available samples
    total_samples = pred.shape[0]
    num_plots = min(100, total_samples)

    # Determine squarish grid
    ncols = int(np.ceil(np.sqrt(num_plots)))
    nrows = int(np.ceil(num_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    fig.tight_layout(pad=2.0)

    # If only one row and one column, axes is not iterable -> make it so
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.expand_dims(axes, axis=0 if nrows == 1 else 1)

    for idx in range(nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        if idx >= num_plots:
            ax.axis('off')
            continue

        # Plot prediction
        ax.plot(x, pred[idx][0], label='Prediction', color='orange')

        # Plot ground truth
        ax.plot(x, truth[idx][0], linestyle='--', label='Ground Truth', color='green')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Sample {idx}")

    # Add single legend outside subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.show()