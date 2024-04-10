# tests/test_config_parsing.py
import sys
import os

import configparser
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestConfigParsing(unittest.TestCase):
    def test_config_parsing(self):
        config = configparser.ConfigParser()
        config.read('../config/node.conf')

        # Assert the configuration is read correctly
        self.assertEqual(config['DEFAULT']['csv_path'], './data/cell_data.csv')
        self.assertEqual(int(config['DEFAULT']['input_dim']), 96)
        self.assertEqual(int(config['DEFAULT']['which_cell']), 0)
        self.assertEqual(int(config['DEFAULT']['batch_size']), 32)
        self.assertEqual(config['P2P']['bootnodes'], '/ip4/127.0.0.1/udp/5000/quic-v1/p2p/12D3KooWSYoEJBh6UtfAT8wdepcvH2sjVGUrSFjgsofZwvNWgFPe')
        self.assertEqual(config['P2P']['key_path'], 'node.key')
        self.assertEqual(config['P2P']['topic'], 'model-net')

if __name__ == '__main__':
    unittest.main()
