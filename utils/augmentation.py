import numpy as np
import torch
import scipy.signal
import soxbindings as sox

def apply_augmentations(audio, target, audio_length, sr):
    # drop continguous frames
    if np.random.rand() < 0.05:
        zero_size = int(audio_length*0.1)
        start = np.random.randint(audio.shape[-1] - zero_size - 1)
        stop = start + zero_size
        audio[:,start:stop] = 0
        if target is not None:
            target[:,start:stop] = 0

    # phase inversion
    if np.random.rand() < 0.5:    
        audio = -audio

    # apply a chorus effect
    if np.random.rand() < 0.05:
        tfm = sox.Transformer()        
        tfm.chorus()

        desired_audio_length = audio.size(dim=1)

        audio = tfm.build_array(input_array=audio.squeeze().numpy(), sample_rate_in=sr)
        audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

        audio = audio[:, :desired_audio_length]
        # if desired_audio_length > audio_length:
        #     audio_length = audio.size(dim=1)
        #     difference = desired_audio_length - audio_length
        #     audio = audio[:, :difference]

    # shift targets forward/back max 70ms
    if np.random.rand() < 0.3 and target is not None:
        # in this method we shift each beat and downbeat by a random amount
        max_shift = int(0.045 * sr)

        beat_ind = torch.logical_and(target[0,:] == 1, target[1,:] != 1).nonzero(as_tuple=False) # all beats EXCEPT downbeats
        dbeat_ind = (target[1,:] == 1).nonzero(as_tuple=False)

        # shift just the downbeats
        dbeat_shifts = torch.normal(0.0, max_shift/2, size=(1,dbeat_ind.shape[-1]))
        dbeat_ind += dbeat_shifts.long()

        # now shift the non-downbeats 
        beat_shifts = torch.normal(0.0, max_shift/2, size=(1,beat_ind.shape[-1]))
        beat_ind += beat_shifts.long()

        # ensure we have no beats beyond max index
        beat_ind = beat_ind[beat_ind < target.shape[-1]]
        dbeat_ind = dbeat_ind[dbeat_ind < target.shape[-1]]  

        # now convert indices back to target vector
        shifted_target = torch.zeros(2,target.shape[-1])
        shifted_target[0,beat_ind] = 1
        shifted_target[0,dbeat_ind] = 1 # set also downbeats on first channel
        shifted_target[1,dbeat_ind] = 1

        target = shifted_target

    # apply pitch shifting
    if np.random.rand() < 0.5:
        sgn = np.random.choice([-1,1])
        factor = sgn * np.random.rand() * 8.0     
        tfm = sox.Transformer()
        tfm.pitch(factor)

        desired_audio_length = audio.size(dim=1)

        audio = tfm.build_array(input_array=audio.squeeze().numpy(), 
                                sample_rate_in=sr)

        audio = np.resize(audio, (1, desired_audio_length))
        audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

        audio = audio[:, :desired_audio_length]
        # if desired_audio_length > audio_length:
        #     audio_length = audio.size(dim=1)
        #     difference = desired_audio_length - audio_length
        #     audio[:, difference:] = 0

    # apply a lowpass filter
    if np.random.rand() < 0.1:
        cutoff = (np.random.rand() * 4000) + 4000
        sos = scipy.signal.butter(2, 
                                  cutoff, 
                                  btype="lowpass", 
                                  fs=sr, 
                                  output='sos')
        audio_filtered = scipy.signal.sosfilt(sos, audio.numpy())
        audio = torch.from_numpy(audio_filtered.astype('float32'))

    # apply a highpass filter
    if np.random.rand() < 0.1:
        cutoff = (np.random.rand() * 1000) + 20
        sos = scipy.signal.butter(2, 
                                  cutoff, 
                                  btype="highpass", 
                                  fs=sr, 
                                  output='sos')
        audio_filtered = scipy.signal.sosfilt(sos, audio.numpy())
        audio = torch.from_numpy(audio_filtered.astype('float32'))

    # apply a compressor effect
    if np.random.rand() < 0.15:
        attack = (np.random.rand() * 0.300) + 0.005
        release = (np.random.rand() * 1.000) + 0.3
        tfm = sox.Transformer()        
        tfm.compand(attack_time=attack, decay_time=release)
        audio = tfm.build_array(input_array=audio.squeeze().numpy(), 
                                sample_rate_in=sr)
        audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

    # apply an EQ effect
    if np.random.rand() < 0.15:
        freq = (np.random.rand() * 8000) + 60
        q = (np.random.rand() * 7.0) + 0.1
        g = np.random.normal(0.0, 6)  
        tfm = sox.Transformer()        
        tfm.equalizer(frequency=freq, width_q=q, gain_db=g)
        audio = tfm.build_array(input_array=audio.squeeze().numpy(), 
                                sample_rate_in=sr)
        audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

    # add white noise
    if np.random.rand() < 0.05:
        wn = (torch.rand(audio.shape) * 2) - 1
        g = 10**(-(np.random.rand() * 20) - 12)/20
        audio = audio + (g * wn)

    # apply nonlinear distortion 
    if np.random.rand() < 0.2:   
        g = 10**((np.random.rand() * 12)/20)   
        audio = torch.tanh(audio)    

    # normalize the audio
    audio /= audio.float().abs().max()

    return audio, target