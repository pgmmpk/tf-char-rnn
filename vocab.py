from __future__ import print_function

import collections
import codecs


class Vocab:
    
    PAD = '\x00'
    GO  = '\x01'
    EOS = '\x02'
    UNK = '\x03'
    
    PAD_ID = 0
    GO_ID  = 1
    EOS_ID = 2
    UNK_ID = 3
    
    START_VOCAB = [PAD, GO, EOS, UNK]
    
    ''' Maps between character and character id. Computations in TF model are made in terms of character ids.
    This class connects TF world with human world by providing means to convert character ids to characters and back.
    '''

    def __init__(self, chars):
        self._chars = chars
        self._decoder = dict(enumerate(list(chars)))
        self._encoder = {b:a for a,b in self._decoder.items()}
        
        assert len(self._encoder) == len(self._decoder)
        assert len(self._encoder) == len(self._chars)

        assert self._decoder[self.PAD_ID] == self.PAD
        assert self._encoder[self.PAD] == self.PAD_ID
        assert self._decoder[self.GO_ID] == self.GO
        assert self._encoder[self.GO] == self.GO_ID
        assert self._decoder[self.EOS_ID] == self.EOS
        assert self._encoder[self.EOS] == self.EOS_ID
        assert self._decoder[self.UNK_ID] == self.UNK
        assert self._encoder[self.UNK] == self.UNK_ID

    def encode(self, c):
        """ char to integer character id """
        return self._encoder.get(c, self.UNK_ID)

    def decode(self, cid):
        """ character id to character """
        return self._decoder.get(cid, self.UNK)

    @property
    def size(self):
        return len(self._chars)

    def __len__(self):
        return self.size

    @classmethod
    def from_data(cls, data, vocab_size):
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        chars = cls.START_VOCAB + [x[0] for x in count_pairs]
        chars = chars[:vocab_size]
        
        if len(chars) < len(set(chars)):
            raise RuntimeError('Data contains symbols allocated as special control symbols! Please change control symbols character value to avoid conflict')

        return Vocab(chars)

    def to_array(self):
        """ saves Vocab to array (useful to initialize TensorFlow Variable for the purpose of saving it to the model file as part of the graph) """
        return [ord(c) for c in self._chars]
    
    @classmethod
    def from_array(cls, array):
        """ restores Vocab from array (useful to create Vocab instance from TensorFlow variable after reading model graph from disk) """
        chars = ''.join(unichr(x) for x in array)

        return Vocab(chars)

    def save(self, filename):
        with codecs.open(filename, 'w', 'utf-8') as f:
            for cid in range(len(self)):
                f.write(str(ord(self.decode(cid))) + '\n')
            
    @classmethod
    def load(cls, filename):
        
        chars = []
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                line = line.strip('\n')
                c = unichr(int(line))
                assert len(c) == 1
                chars.append(c)

        return Vocab(chars)


if __name__ == '__main__':
    v = Vocab.from_data('Hello')
    print(v.size)

    encoded = [v.encode(x) for x in 'Hello']
    print(encoded)

    v = Vocab.from_array(v.to_array())

    decoded = [v.decode(x) for x in encoded]
    print(decoded)
