# History
- **v0.2.5**：20221127 对抗训练从compile转为使用Callback来实现，修复1.7.1版本兼容bug, uie模型内置
- **v0.2.4**：20221120 删除SpTokenizer基类中的rematch, 增加deberta_v2模型
- **v0.2.3**：20221023 虚拟对抗VAT在多个ouput时支持指定，把Trainer抽象到[torch4keras](https://github.com/Tongjilibo/torch4keras)中，修复DP和DDP出现resume_epoch不存在的bug, tokenizer的never_split去除None, transformer_xl的bug, 增加gradient_checkpoint
- **v0.2.2**：20220922 修复t5的norm_mode问题，允许hidden_size不整除num_attention_heads，支持多个schedule(如同时ema+warmup)
- **v0.2.1**：20220905 兼容torch<=1.7.1的torch.div无rounding_mode，增加自定义metrics，支持断点续训，增加默认Logger和Tensorboard日志
- **v0.2.0**：20220823 兼容torch<1.9.0的缺失take_along_dim，修复bart中位置向量514的问题，修复Sptokenizer对符号不转换，打印Epoch开始的时间戳，增加parallel_apply
- **v0.1.9**：20220808 增加mixup/manifold_mixup/temporal_ensembling策略，修复pgd策略param.grad为空的问题，修改tokenizer支持批量
- **v0.1.8**：20220717 修复原来CRF训练中loss陡增的问题，修复xlnet的token_type_ids输入显存占用大的问题
- **v0.1.7**：20220710 增加EarlyStop，CRF中自带转bool类型
- **v0.1.6**：20220605 增加transformer_xl、xlnet、t5_pegasus模型，prompt、预训练等示例，支持增加embedding输入，EMA策略，修复tokenizer和sinusoid的bug
- **v0.1.5**：20220504 增加GAU-alpha，混合梯度，梯度裁剪，单机多卡(DP、DDP)
- **v0.1.4**：20220421 增加了VAT，修复了linux下apply_embedding返回项有问题的情况
- **v0.1.3**：20220409 初始版本
