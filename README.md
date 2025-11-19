## 1. Create and activate a virtual environment

- Create a new virtual environment with Python 3.9：
    conda create -n PROTOCOL python=3.9
- Activate the environment：
    conda activate PROTOCOL



## 2. Install dependencies

- Install required packages from requirements.txt：
    pip install -r Requirements.txt
- Install packages individually as needed



## 3. Modify Transformer.py to obtain G
- Step 1: Find the path of transformer.py in the PROTOCOL conda environment.
- Step 2: Enter the internal code of TransformerEncoderLayer.
- Step 3: Locate Code 1 and replace it with Code 2.
        Code1: 
        -------------------------------------------------------------------------------------
                x = src
                if self.norm_first:
                    x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
                    x = x + self._ff_block(self.norm2(x))
                else:
                    x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
                    x = self.norm2(x + self._ff_block(x))

                return x

            # self-attention block
            def _sa_block(self, x: Tensor,
                        attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
                x = self.self_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False, is_causal=is_causal)[0]
                return self.dropout1(x)
        -------------------------------------------------------------------------------------
        Code2:
        -------------------------------------------------------------------------------------
        x = src
                first,  second = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                if self.norm_first:
                    x = x + first
                    x = x + self._ff_block(self.norm2(x))
                else:
                    x = self.norm1(x + first)
                    x = self.norm2(x + self._ff_block(x))

                return x,second

            # self-attention block
            def _sa_block(self, x: Tensor,
                        attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
                x,y = self.self_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
                return self.dropout1(x),y
        -------------------------------------------------------------------------------------
- Step 4: Enter the 'MutiheadAttention' function (Code3) and locate Code 4 to set the 'need_weights' parameter to True.
        Code3:
        -------------------------------------------------------------------------------------
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        -------------------------------------------------------------------------------------
        Code4:
        -------------------------------------------------------------------------------------
        ...
        key_padding_mask=key_padding_mask, need_weights=True,
                        attn_mask=attn_mask,
                        use_separate_proj_weight=True,
                        q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                        v_proj_weight=self.v_proj_weight,
                        average_attn_weights=average_attn_weights,
                        is_causal=is_causal)
                else:
                    attn_output, attn_output_weights = F.multi_head_attention_forward(
                        query, key, value, self.embed_dim, self.num_heads,
                        self.in_proj_weight, self.in_proj_bias,
                        self.bias_k, self.bias_v, self.add_zero_attn,
                        self.dropout, self.out_proj.weight, self.out_proj.bias,
                        training=self.training,
                        key_padding_mask=key_padding_mask,
                        need_weights=True,
        ...
        -------------------------------------------------------------------------------------

## 4.Usage
- Before run, please carefully read  ''3. Modify Transformer.py to obtain G'', and refer to the steps inside it to modify the code in order to obtain G.

- test:
```bash
python test.py
```

- train:
```bash
python train.py
```

## References
If you find the code useful for your research, please consider citing
```bib
@inproceedings{xue2025protocolpartialoptimaltransportenhanced,
      title={PROTOCOL: Partial Optimal Transport-enhanced Contrastive Learning for Imbalanced Multi-view Clustering}, 
      author={Xuqian Xue and Yiming Lei and Qi Cai and Hongming Shan and Junping Zhang},
      year={2025},
      booktitle = {Proceedings of the 42th International Conference on Machine Learning}
}

```
## Acknowledgements

This implementation is based on / inspired by:

- [https://github.com/rhfeiyang/PPOT]








