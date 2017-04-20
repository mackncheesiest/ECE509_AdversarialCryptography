"""
    Defines a text<->binary encoder class that always results in printable text rather than, like, weird ascii values
"""


class Encoder:
    def __init__(self):
        self.dict = {}
        # Note: this was originally a, b, ..., z, period, !, ?, comma, ;, space
        # But it was ending up with terrible results with trained networks,
        # And I think it's because the distribution of the data when we translate from English this way
        # is almost nothing like the uniform distribution that generateData draws from
        # i.e. otherwise vowels like 'a' and 'e' are going to skew the encoded data towards containing tons of zeroes.
        self.dict['s'] = '00000'
        self.dict['x'] = '00001'
        self.dict['n'] = '00010'
        self.dict['g'] = '00011'
        self.dict['z'] = '00100'
        self.dict['?'] = '00101'
        self.dict['b'] = '00110'
        self.dict[';'] = '00111'
        self.dict['m'] = '01000'
        self.dict['u'] = '01001'
        self.dict['j'] = '01010'
        self.dict['c'] = '01011'
        self.dict['!'] = '01100'
        self.dict['k'] = '01101'
        self.dict['o'] = '01110'
        self.dict['a'] = '01111'
        self.dict['l'] = '10000'
        self.dict[' '] = '10001'
        self.dict['d'] = '10010'
        self.dict['i'] = '10011'
        self.dict['e'] = '10100'
        self.dict['q'] = '10101'
        self.dict['t'] = '10110'
        self.dict['f'] = '10111'
        self.dict['y'] = '11000'
        self.dict['w'] = '11001'
        self.dict['.'] = '11010'
        self.dict[','] = '11011'
        self.dict['h'] = '11100'
        self.dict['v'] = '11101'
        self.dict['p'] = '11110'
        self.dict['r'] = '11111'
        
        self.inv_dict = {val: key for key, val in self.dict.items()}
        
        self.validChars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
                           'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
                           'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                           'y', 'z', '.', '!', '?', ',', ';', ' ']
        
    def encode(self, inputStr):
        try:
            return ''.join([self.dict[c] for c in inputStr.lower()])
        except KeyError as e:
            print("Invalid character: " + str(e.args[0]))
            print("These are the valid characters that may be encoded:\n" + str(self.validChars))
            
    # Inverse dictionary mappings: http://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping
    def decode(self, inputStr):
        try:
            # Need to group the input string into groups of 5 for decoding
            # http://stackoverflow.com/questions/9475241/split-python-string-every-nth-character
            splitInputStr = [inputStr[i:i+5] for i in range(0, len(inputStr), 5)]
            return ''.join([self.inv_dict[c] for c in splitInputStr])
        except KeyError:
            print("These are the valid characters that may be decoded:\n" + str(self.validChars))
        
    