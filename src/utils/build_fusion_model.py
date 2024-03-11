import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout, Input, Multiply, LayerNormalization, MultiHeadAttention, Concatenate
from tensorflow.keras.models import Model
from utils.dcp_layer import DarkChannelPriorLayerV2
from utils.edge_layer import EdgeDetectionLayerV2
from utils.densenet121 import densenet
from utils.self_entropy import SelfEntropyLayerV2

def build_model(config):
    num_channels = 3
    number_of_modalities = 0
    edge_layer = config.edge
    entropy_layer = config.entropy
    dark_channel_layer = config.dcp
    rgb_layer = config.rgb
    depth_layer = config.depth
    normal_layer = config.normal

    inputs = Input(shape=(config.img_height, config.img_width, num_channels))
    
    
    x = inputs
    embeddings = []

    edge_output = None
    entropy_output = None
    dark_channel_output = None
    depth_output = None
    normal_output = None

    x_edge = None
    x_depth = None
    x_normal = None
    x_entropy = None
    x_dark_channel = None
    x_rgb = None

    base_model_rgb = None
    base_model_edge = None
    base_model_entropy = None
    base_model_dark_channel = None
    base_model_depth = None
    base_model_normal = None

    # Define base models for each modality
    if rgb_layer:
        base_model_rgb = densenet(input_shape=(config.img_height, config.img_width, num_channels))
        base_model_rgb.trainable = config.trainable_epochs == 0
        x_rgb = base_model_rgb(x)
        x_rgb = GlobalAveragePooling2D()(x_rgb)
        x_rgb = LayerNormalization()(x_rgb)
        embeddings.append(x_rgb)
        number_of_modalities += 1

    if edge_layer:
        base_model_edge = densenet(input_shape=(config.img_height, config.img_width, num_channels))
        base_model_edge.trainable = config.trainable_epochs == 0
        x_edge = EdgeDetectionLayerV2()(inputs)
        x_edge = base_model_edge(x_edge)
        x_edge = GlobalAveragePooling2D()(x_edge)
        x_edge = LayerNormalization()(x_edge)
        embeddings.append(x_edge)
        number_of_modalities += 1

    if entropy_layer:
        base_model_entropy = densenet(input_shape=(config.img_height, config.img_width, num_channels))
        base_model_entropy.trainable = config.trainable_epochs == 0
        x_entropy = SelfEntropyLayerV2()(inputs)
        x_entropy = base_model_entropy(x_entropy)
        x_entropy = GlobalAveragePooling2D()(x_entropy)
        x_entropy = LayerNormalization()(x_entropy)
        embeddings.append(x_entropy)
        number_of_modalities += 1

    if dark_channel_layer:
        base_model_dark_channel = densenet(input_shape=(config.img_height, config.img_width, num_channels))
        base_model_dark_channel.trainable = config.trainable_epochs == 0
        x_dark_channel = DarkChannelPriorLayerV2()(inputs)
        x_dark_channel = base_model_dark_channel(x_dark_channel)
        x_dark_channel = GlobalAveragePooling2D()(x_dark_channel)
        x_dark_channel = LayerNormalization()(x_dark_channel)
        embeddings.append(x_dark_channel)
        number_of_modalities += 1

    if depth_layer:
        depth_inputs = Input(shape=(config.img_height, config.img_width, num_channels))
        base_model_depth = densenet(input_shape=(config.img_height, config.img_width, num_channels))
        base_model_depth.trainable = config.trainable_epochs == 0
        x_depth = base_model_depth(depth_inputs)
        x_depth = GlobalAveragePooling2D()(x_depth)
        x_depth = LayerNormalization()(x_depth)
        embeddings.append(x_depth)
        number_of_modalities += 1



    if normal_layer:
        normal_inputs = Input(shape=(config.img_height, config.img_width, num_channels))
        base_model_normal = densenet(input_shape=(config.img_height, config.img_width, num_channels))
        base_model_normal.trainable = config.trainable_epochs == 0
        x_normal = base_model_normal(normal_inputs)
        x_normal = GlobalAveragePooling2D()(x_normal)
        # Normalize each modality separatly, this assure that no modality dominate the results before feeding it to the mlp head
        x_normal = LayerNormalization()(x_normal)
        embeddings.append(x_normal)
        number_of_modalities += 1

    # Concatenate normalized embeddings
    xc = Concatenate(axis=-1)(embeddings)


    xc_shape = xc.shape

    # Flattening
    xc = Flatten()(xc)  # Flatten 

    # Dense layers
    xc = Dense(256, activation='relu')(xc)
    xc = Dropout(rate=0.2)(xc)
    xc = Dense(256, activation='relu')(xc)
    xxc = Dropout(rate=0.2)(xc)
    predictions = Dense(config.num_classes, activation='softmax')(xc)

    model_inputs = []
    if rgb_layer or edge_layer:
        model_inputs.append(inputs)
    if depth_layer:
        model_inputs.append(depth_inputs)
    if normal_layer:
        model_inputs.append(normal_inputs)
    # RGB, Edge, Entropy, DCP, Depth, Normal 
    model = Model(inputs=model_inputs, outputs=predictions)
    return model
