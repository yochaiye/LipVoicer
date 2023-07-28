from ASR.nnet import AudioEfficientConformerInterCTC
from ASR.nnet import CTCLoss, CTCBeamSearchDecoder
import sentencepiece as spm

asr_guidance_net = AudioEfficientConformerInterCTC(interctc_blocks=[], T=400, beta_0=0.0001, beta_T=0.02)
checkpoint_ao = "ASR/callbacks/LRS23/AO/EffConfCTC/checkpoints_epoch_4_step_499.ckpt"
asr_guidance_net.compile(losses=CTCLoss(zero_infinity=True, assert_shorter=False), loss_weights=None)
asr_guidance_net = asr_guidance_net.cuda()
asr_guidance_net.load(checkpoint_ao)
asr_guidance_net.eval()
tokenizer_path = "ASR/media/tokenizerbpe256.model"
tokenizer = spm.SentencePieceProcessor(tokenizer_path)  # for converting text to tokens
ngram_path = "ASR/media/6gram_lrs23.arpa"
neural_config_path = "ASR/configs/LRS23/LM/GPT-Small-demo.py"
neural_checkpoint = "checkpoints_epoch_10_step_2860.ckpt"
decoder = CTCBeamSearchDecoder(
    tokenizer_path=tokenizer_path,
    ngram_path=ngram_path,
    neural_config_path=neural_config_path,
    neural_checkpoint=neural_checkpoint,
)  # For converting tokens to text at the ASR output