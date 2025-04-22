commands :
python train.py --data ./dataset/malayalam_chars
python test.py --data ./dataset/malayalam_chars --model_path results-20250414-112908/best_weights.pt

Results:
04/14 11:32:18 AM Epoch 001: Train Loss: 0.1194 | Train Acc: 48.97% | Val Acc: 57.65%
E:\sem4\ML\project\ADARTS project\ADARTS project\ADARTS\utils.py:102: UserWarning: The torch.cuda._DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=_, device='cuda') to create tensors. (Triggered internally at C:\actions-runner_work\pytorch\pytorch\pytorch\torch\csrc\tensor\python*tensor.cpp:80.)
mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli*(keep_prob))
04/14 11:33:38 AM Epoch 002: Train Loss: 0.0569 | Train Acc: 74.33% | Val Acc: 81.14%
04/14 11:34:59 AM Epoch 003: Train Loss: 0.0411 | Train Acc: 81.72% | Val Acc: 84.53%
04/14 11:36:18 AM Epoch 004: Train Loss: 0.0349 | Train Acc: 83.84% | Val Acc: 86.42%
04/14 11:37:36 AM Epoch 005: Train Loss: 0.0301 | Train Acc: 86.24% | Val Acc: 90.75%
04/14 11:38:54 AM Epoch 006: Train Loss: 0.0277 | Train Acc: 86.84% | Val Acc: 89.08%
04/14 11:40:13 AM Epoch 007: Train Loss: 0.0255 | Train Acc: 88.39% | Val Acc: 91.24%
04/14 11:40:13 AM Epoch 007: Train Loss: 0.0255 | Train Acc: 88.39% | Val Acc: 91.24%
04/14 11:40:13 AM Epoch 007: Train Loss: 0.0255 | Train Acc: 88.39% | Val Acc: 91.24%
04/14 11:41:31 AM Epoch 008: Train Loss: 0.0244 | Train Acc: 88.53% | Val Acc: 89.12%
04/14 11:40:13 AM Epoch 007: Train Loss: 0.0255 | Train Acc: 88.39% | Val Acc: 91.24%
04/14 11:41:31 AM Epoch 008: Train Loss: 0.0244 | Train Acc: 88.53% | Val Acc: 89.12%
04/14 11:41:31 AM Epoch 008: Train Loss: 0.0244 | Train Acc: 88.53% | Val Acc: 89.12%
04/14 11:42:49 AM Epoch 009: Train Loss: 0.0227 | Train Acc: 89.07% | Val Acc: 92.55%
04/14 11:42:49 AM Epoch 009: Train Loss: 0.0227 | Train Acc: 89.07% | Val Acc: 92.55%
04/14 11:44:08 AM Epoch 010: Train Loss: 0.0221 | Train Acc: 89.52% | Val Acc: 93.90%
04/14 11:44:08 AM Epoch 010: Train Loss: 0.0221 | Train Acc: 89.52% | Val Acc: 93.90%
04/14 11:45:26 AM Epoch 011: Train Loss: 0.0204 | Train Acc: 90.42% | Val Acc: 93.78%
04/14 11:46:44 AM Epoch 012: Train Loss: 0.0189 | Train Acc: 90.96% | Val Acc: 91.57%
04/14 11:48:02 AM Epoch 013: Train Loss: 0.0198 | Train Acc: 90.45% | Val Acc: 92.96%
04/14 11:49:19 AM Epoch 014: Train Loss: 0.0186 | Train Acc: 91.36% | Val Acc: 94.93%
04/14 11:50:37 AM Epoch 015: Train Loss: 0.0174 | Train Acc: 91.89% | Val Acc: 94.97%
04/14 11:51:56 AM Epoch 016: Train Loss: 0.0173 | Train Acc: 91.75% | Val Acc: 93.82%
04/14 11:53:13 AM Epoch 017: Train Loss: 0.0176 | Train Acc: 91.79% | Val Acc: 94.64%
04/14 11:54:31 AM Epoch 018: Train Loss: 0.0161 | Train Acc: 91.92% | Val Acc: 95.13%
04/14 11:55:48 AM Epoch 019: Train Loss: 0.0166 | Train Acc: 91.92% | Val Acc: 94.93%
04/14 11:57:06 AM Epoch 020: Train Loss: 0.0167 | Train Acc: 92.07% | Val Acc: 95.83%
04/14 11:58:26 AM Epoch 021: Train Loss: 0.0154 | Train Acc: 92.32% | Val Acc: 96.64%
04/14 11:59:44 AM Epoch 022: Train Loss: 0.0154 | Train Acc: 92.49% | Val Acc: 95.79%
04/14 12:01:03 PM Epoch 023: Train Loss: 0.0145 | Train Acc: 92.66% | Val Acc: 97.05%
04/14 12:02:20 PM Epoch 024: Train Loss: 0.0158 | Train Acc: 92.61% | Val Acc: 95.50%
04/14 12:03:37 PM Epoch 025: Train Loss: 0.0163 | Train Acc: 91.84% | Val Acc: 95.95%
04/14 12:05:33 PM Epoch 026: Train Loss: 0.0149 | Train Acc: 92.92% | Val Acc: 95.34%
04/14 12:07:57 PM Epoch 027: Train Loss: 0.0145 | Train Acc: 93.14% | Val Acc: 95.95%
04/14 12:09:15 PM Epoch 028: Train Loss: 0.0148 | Train Acc: 92.57% | Val Acc: 96.64%
04/14 12:10:33 PM Epoch 029: Train Loss: 0.0135 | Train Acc: 93.39% | Val Acc: 96.19%
04/14 12:11:52 PM Epoch 030: Train Loss: 0.0144 | Train Acc: 92.89% | Val Acc: 96.36%
04/14 12:13:10 PM Epoch 031: Train Loss: 0.0140 | Train Acc: 92.82% | Val Acc: 95.38%
04/14 12:14:28 PM Epoch 032: Train Loss: 0.0153 | Train Acc: 92.55% | Val Acc: 96.69%
04/14 12:15:47 PM Epoch 033: Train Loss: 0.0144 | Train Acc: 93.03% | Val Acc: 96.69%
04/14 12:17:05 PM Epoch 034: Train Loss: 0.0141 | Train Acc: 92.74% | Val Acc: 96.73%
04/14 12:18:23 PM Epoch 035: Train Loss: 0.0151 | Train Acc: 92.80% | Val Acc: 97.01%
04/14 12:19:41 PM Epoch 036: Train Loss: 0.0134 | Train Acc: 93.50% | Val Acc: 96.97%
04/14 12:21:00 PM Epoch 037: Train Loss: 0.0139 | Train Acc: 93.08% | Val Acc: 97.18%
04/14 12:22:19 PM Epoch 038: Train Loss: 0.0147 | Train Acc: 92.89% | Val Acc: 96.93%
04/14 12:23:37 PM Epoch 039: Train Loss: 0.0146 | Train Acc: 92.80% | Val Acc: 97.05%
04/14 12:24:55 PM Epoch 040: Train Loss: 0.0144 | Train Acc: 93.05% | Val Acc: 97.26%
04/14 12:26:13 PM Epoch 041: Train Loss: 0.0130 | Train Acc: 93.59% | Val Acc: 97.18%
04/14 12:27:31 PM Epoch 042: Train Loss: 0.0147 | Train Acc: 92.92% | Val Acc: 96.97%
04/14 12:28:48 PM Epoch 043: Train Loss: 0.0144 | Train Acc: 92.91% | Val Acc: 97.59%
04/14 12:30:06 PM Epoch 044: Train Loss: 0.0152 | Train Acc: 92.60% | Val Acc: 96.69%
04/14 12:31:24 PM Epoch 045: Train Loss: 0.0143 | Train Acc: 93.05% | Val Acc: 97.59%
04/14 12:32:41 PM Epoch 046: Train Loss: 0.0147 | Train Acc: 93.30% | Val Acc: 96.97%
04/14 12:33:59 PM Epoch 047: Train Loss: 0.0139 | Train Acc: 93.28% | Val Acc: 97.46%
04/14 12:35:17 PM Epoch 048: Train Loss: 0.0155 | Train Acc: 92.94% | Val Acc: 97.71%
04/14 12:36:35 PM Epoch 049: Train Loss: 0.0152 | Train Acc: 92.60% | Val Acc: 97.26%
04/14 12:37:52 PM Epoch 050: Train Loss: 0.0146 | Train Acc: 92.82% | Val Acc: 97.67%
04/14 12:37:52 PM Final Best Validation Accuracy: 97.71%
(base) PS E:\sem4\ML\project\ADARTS project\ADARTS project\ADARTS>
