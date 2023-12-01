import numpy as np

class DataLoader:
    def __init__(self, data, shuffle = False, batch_size = 1):

        self.batch_size = batch_size
        self.shuffle = shuffle
        if isinstance(data, str):
            data = np.load(data)
            self.X = data[:,:-1]
            self.y = data[:,-1]
        else:
            self.X = data[0]
            self.y = data[1]

        self.size = self.X.shape[0]
        self.shape = self.X.shape
        

    def __iter__(self):
        if self.shuffle:
            # Shuffle the data if specified
            self._shuffle_data()

        # Create an iterator for batches
        self.current_index = 0
        return self

    def __next__(self):
        # Check if batch sizeis 1 or not
        if self.batch_size == 1:
            if self.current_index >= self.size:
                raise StopIteration
            
            val = self[self.current_index]

            self.current_index += 1
            
            return val

        # Check if we have reached the end of the data
        if self.current_index + self.batch_size > self.size:
            raise StopIteration

        # Get the batch
        batch_x, batch_y = self.X[self.current_index : self.current_index + self.batch_size], self.y[self.current_index : self.current_index + self.batch_size]

        # Move the index to the next batch
        self.current_index += self.batch_size

        return batch_x, batch_y

    def _shuffle_data(self):
        # Shuffle the data
        order = np.random.permutation(self.size)
        self.X = self.X[order]
        self.y = self.y[order]

    def __getitem__(self, inx: int):
        # If index is an integer
        if isinstance(inx, int):
            if inx < self.size:
                return self.X[inx], self.y[inx]
            else:
                raise IndexError
            
        # If index is a slice
        if isinstance(inx, slice):
            if inx.start < self.size:
                return self.X[inx], self.y[inx]
            else:
                raise IndexError
            
    def __len__(self):
        return self.size
        
    def __repr__(self):
        return f"Data with shape of {self.shape}, shuffle = {self.shuffle}, batch_size = {self.batch_size}"
    