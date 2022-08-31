import sys
from importlib import import_module

import torch
from flask import Flask, render_template, request

sys.path.append('models')
sys.path.append('src')
from models import bert
from src.global_config import Global_Config

app = Flask(__name__)


@app.route('/')
def index():
    # return "hello"
    return render_template('index.html')


@app.route('/text_predict', methods=['POST'])
def text_predict():
    classes = ['Negative：消极评论', 'Positive：积极评论', 'Neutral：中立评论']
    PAD, CLS = '[PAD]', '[CLS]'

    def dataloader(input, config, pad_size=32):
        contents = []
        tokenizer = config.tokenizer.from_pretrained('./bert_pretrain')
        token = tokenizer.tokenize(input)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = tokenizer.convert_tokens_to_ids(token)
        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        contents.append((token_ids, seq_len, mask))

        x = torch.LongTensor([_[0] for _ in contents])  # .to(config.device)
        seq_len = torch.LongTensor([_[1] for _ in contents])  # .to(config.device)
        mask = torch.LongTensor([_[2] for _ in contents])  # .to(config.device)
        return x, seq_len, mask

    def predict(x, seq_len, mask):
        model.eval()
        with torch.no_grad():
            output = model([x, seq_len, mask])
            predic = torch.max(output.data, 1)[1].cpu().numpy()
            # y = int(np.argmax(model.predict(test), axis=1))
        return classes[predic.item()]

    if request.method == 'GET':
        pass
    else:
        data = request.json  # 获取 JOSN 数据
        data = data.get('content')
        print(data)
        # seg_string = [' '.join(seg_list)]
        x, seq_len, mask = dataloader(data, config)
        label = predict(x, seq_len, mask)
        return "类别：" + label


if __name__ == '__main__':
    g_config = Global_Config('bert', train=False)
    g_config.class_list = [x.strip() for x in open(
        'THUCNews/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
    g_config.num_classes = len(g_config.class_list)
    g_config.save_path = 'THUCNews/saved_dict/bert.ckpt'  # 模型训练结果
    x = import_module('models.' + 'bert')
    config = x.Config('THUCNews')
    model = x.Model(config)
    model.load_state_dict(torch.load(config.save_path))
    # model  # .cuda()
    app.run(host="192.168.12.49", port=5000, debug=False, threaded=False)
