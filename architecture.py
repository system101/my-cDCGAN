
class Architecture:

    img_size = 28
    num_cat = 27
    z_size = 500 # 100

    # Generator layer config
    layers_g = [
        # {
        #     'filters':1024,
        #     'kernel_size':[4,4],
        #     'strides':[1,1],
        #     'padding':'valid'
        # },
        # {
        #     'filters':512,
        #     'kernel_size':[4,4],
        #     'strides':[2,2],
        #     'padding':'same'
        # },
        # {
        #     'filters':256,
        #     'kernel_size':[4,4],
        #     'strides':[2,2],
        #     'padding':'same'
        # },
        # {
        #     'filters':128,
        #     'kernel_size':[4,4],
        #     'strides':[2,2],
        #     'padding':'same'
        # },
        # {
        #     'filters':1,
        #     'kernel_size':[4,4],
        #     'strides':[2,2],
        #     'padding':'same'
        # }
        {
            'filters':256,
            'kernel_size':[7,7],
            'strides':[1,1],
            'padding':'valid'
        },
        {
            'filters':128,
            'kernel_size':[5,5],
            'strides':[2,2],
            'padding':'same'
        },
        {
            'filters':1,
            'kernel_size':[5,5],
            'strides':[2,2],
            'padding':'same'
        }
    ]

    # Discriminator layer config
    layers_d = [
        # {
        #     'filters': 128,
        #     'kernel_size':[4,4],
        #     'strides':[2,2],
        #     'padding':'same'
        # },
        # {
        #     'filters': 256,
        #     'kernel_size':[4,4],
        #     'strides':[2,2],
        #     'padding':'same'
        # },
        # {
        #     'filters': 512,
        #     'kernel_size':[4,4],
        #     'strides':[2,2],
        #     'padding':'same'
        # },
        # {
        #     'filters': 1024,
        #     'kernel_size':[4,4],
        #     'strides':[2,2],
        #     'padding':'same'
        # },
        # {
        #     'filters': 1,
        #     'kernel_size': [4,4],
        #     'strides':[1,1],
        #     'padding':'valid'
        # }

        {
            'filters': 128,
            'kernel_size': [5,5],
            'strides':[2,2],
            'padding':'same'
        },
        {
            'filters': 256,
            'kernel_size':[5,5],
            'strides':[2,2],
            'padding':'same'
        },
        {
            'filters': 1,
            'kernel_size':[7,7],
            'strides':[1,1],
            'padding':'valid'
        }
        
    ]
    

