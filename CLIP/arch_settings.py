rn50_arch_dict = {"embed_dim":1024,
                 "image_resolution":224,
                 "vision_layers":[3,4,6,3],
                 "vision_width":2048,
                 "vision_patch_size":0,
                 "context_length":78,
                 "vocab_size":49408,
                 "transformer_heads":12,
                 "transformer_width":512,
                 "transformer_layers":8}


rn50_opt_dict = {"lr":5e-4,
                    "adam_beta2":0.999,
                    "adam_eps":10e-8}


vit_l_14_arch_dict = {"embed_dim":768,
                     "image_resolution":224,
                     "vision_layers":24,
                     "vision_width":1024,
                     "vision_patch_size":14,
                     "context_length":78,
                     "vocab_size":49408,
                     "transformer_heads":12,
                     "transformer_width":768,
                     "transformer_layers":12}

vit_l_14_opt_dict = {"lr":4e-4,
                    "adam_beta2":0.98,
                    "adam_eps":10e-6}


model_to_settings = { "ResNet50":(rn50_arch_dict, rn50_opt_dict),
                   "ViT-L/14":(vit_l_14_arch_dict, vit_l_14_opt_dict),
                   }
