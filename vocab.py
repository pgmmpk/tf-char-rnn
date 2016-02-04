from __future__ import print_function
    
import collections


class Vocab:
    ''' Maps between character and character id. Computations in TF model are made in terms of character ids. 
    This class connects TF world with human world by providing means to conver character ids to characters and back.
    '''
    
    def __init__(self, chars):
        self._chars = chars
        self._decoder = dict(enumerate(chars))
        self._encoder = {b:a for a,b in self._decoder.items()}
                         
        self._default_char = chars[-1]
        self._default_char_id = self._encoder[self._default_char]

        assert self._default_char_id == len(chars) - 1

    def encode(self, c):
        """ char to integer id """
        return self._encoder.get(c, self._default_char_id)
    
    def decode(self, cid):
        return self._decoder.get(cid, self._default_char)
    
    @property
    def size(self):
        return len(self._encoder)
    
    def __len__(self):
        return self.size
    
    @classmethod
    def from_data(cls, data, size_limit=None):
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        if size_limit:
            count_pairs = count_pairs[:size_limit]
        
        chars, _ = list(zip(*count_pairs))

        return Vocab(chars)

    def to_array(self):
        """ saves Vocab to array (useful to initialize TensorFlow Variable for the purpose of saving it to the model file as part of the graph) """
        return [ord(c) for c in self._chars]
    
    @classmethod
    def from_array(cls, array):
        """ restores Vocab from array (useful to create Vocab instance from TensorFlow variable after reading model graph from disk) """
        chars = ''.join(unichr(x) for x in array)
        
        return Vocab(chars)


if __name__ == '__main__':
    v = Vocab.from_data('Hello')
    print(v.size)
    
    encoded = [v.encode(x) for x in 'Hello']
    print(encoded)
    
    v = Vocab.from_array(v.to_array())
    
    decoded = [v.decode(x) for x in encoded]
    print(decoded)

    