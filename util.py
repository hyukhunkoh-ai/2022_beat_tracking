def pad(x, max_length=12.8):
        """
        Pad inputs (on left/right and up to predefined length or max length in the batch)
        Args:
            x: input audio sequence length in seconds
            max_length: maximum length of the returned list and optionally padding length (see below)
        """

        attention_mask = np.ones(x*22050, dtype=np.int32)

        difference = (max_length - x)*22050
        attention_mask = np.pad(attention_mask, (0, difference))
        padding_shape = (0, difference)
        padded_x = np.pad(x, padding_shape, "constant", constant_values=-1)

        return padded_x, attention_mask
