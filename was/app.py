import pandas as pd
from flask_cors import CORS
from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
import torch

from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib

print('test');
