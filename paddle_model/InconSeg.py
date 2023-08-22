import paddle
import paddle.nn as nn
import paddle.vision.models as models


class InconSeg(nn.Layer):

    def __init__(self, n_class):
        super(InconSeg, self).__init__()

        resnet_raw_model1 = models.resnet152(pretrained=True)
        resnet_raw_model2 = models.resnet152(pretrained=True)
        self.inplanes = 2048

        ########  Thermal ENCODER  ########
        self.encoder_depth_conv1 = nn.Conv2D(1, 64, kernel_size=7, stride=2, padding=3) 
        #self.encoder_depth_conv1.weight.data = paddle.unsqueeze(paddle.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        self.encoder_depth_bn1 = resnet_raw_model1.bn1
        self.encoder_depth_relu = resnet_raw_model1.relu
        self.encoder_depth_maxpool = resnet_raw_model1.maxpool
        self.encoder_depth_layer1 = resnet_raw_model1.layer1
        self.encoder_depth_layer2 = resnet_raw_model1.layer2
        self.encoder_depth_layer3 = resnet_raw_model1.layer3
        self.encoder_depth_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        ########  DECODER  ########
        self.skip_tranform_rgb = nn.Conv2D(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.deconv5_rgb = upbolckV1(cin=2048,cout=1024)
        self.deconv4_rgb = upbolckV1(cin=1024,cout=512)
        self.deconv3_rgb = upbolckV1(cin=512,cout=256)
        self.deconv2_rgb = upbolckV1(cin=256,cout=128)
        self.deconv1_rgb = upbolckV1(cin=128,cout=n_class)

        self.skip_tranform_depth = nn.Conv2D(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.deconv5_depth = upbolckV1(cin=2048,cout=1024)
        self.deconv4_depth = upbolckV1(cin=1024,cout=512)
        self.deconv3_depth = upbolckV1(cin=512,cout=256)
        self.deconv2_depth = upbolckV1(cin=256,cout=128)
        self.deconv1_depth = upbolckV1(cin=128,cout=n_class)

        self.fusion5 = FusionV1()
        self.fusion4 = FusionV1()
        self.fusion3 = FusionV10(in_channel=256,n_class=n_class)
        self.fusion2 = FusionV10(in_channel=128,n_class=n_class)
        self.fusion1 = FusionV10(in_channel=n_class,n_class=n_class)

    def forward(self, input):

        rgb = input[:,:3]
        depth = input[:,3:]

        verbose = False

        # encoder

        ######################################################################

        if verbose: print("rgb.shape original: ", rgb.shape)  # (288, 512)
        if verbose: print("depth.shape original: ", depth.shape)  # (288, 512)
        ######################################################################


        depth = self.encoder_depth_conv1(depth)
        if verbose: print("depth.shape after conv1: ", depth.shape) # (144, 256)
        depth = self.encoder_depth_bn1(depth)
        if verbose: print("depth.shape after bn1: ", depth.shape)  # (144, 256)
        depth = self.encoder_depth_relu(depth)
        if verbose: print("depth.shape after relu: ", depth.shape)  # (144, 256)
        skip_d_1 = depth

        ######################################################################

        depth = self.encoder_depth_maxpool(depth)
        if verbose: print("depth.shape after maxpool: ", depth.shape) # (72, 128)
        depth = self.encoder_depth_layer1(depth)
        if verbose: print("depth.shape after layer1: ", depth.shape) # (72, 128)
        skip_d_2 = depth

        ######################################################################
 
        depth = self.encoder_depth_layer2(depth)
        if verbose: print("depth.shape after layer2: ", depth.shape) # (36, 64)
        skip_d_3 = depth
        ######################################################################

        depth = self.encoder_depth_layer3(depth)
        if verbose: print("depth.shape after layer3: ", depth.shape) # (18, 32)
        skip_d_4 = depth
        ######################################################################

        depth = self.encoder_depth_layer4(depth)
        if verbose: print("depth.shape after layer4: ", depth.shape) # (9, 16)

        ######################################################################

        # decoder

        depth_decoder_out_5 = self.deconv5_depth(depth)
        depth_decoder_out_5 = depth_decoder_out_5+skip_d_4
        if verbose: print("fuse after deconv1: ", depth_decoder_out_5.shape) # (18, 32)

        depth_decoder_out_4 = self.deconv4_depth(depth_decoder_out_5)
        depth_decoder_out_4 = depth_decoder_out_4+skip_d_3
        if verbose: print("fuse after deconv2: ", depth_decoder_out_4.shape) # (36, 64)

        depth_decoder_out_3 = self.deconv3_depth(depth_decoder_out_4)
        depth_decoder_out_3 = depth_decoder_out_3+skip_d_2
        if verbose: print("fuse after deconv3: ", depth_decoder_out_3.shape) # (72, 128)

        depth_decoder_out_2 = self.deconv2_depth(depth_decoder_out_3)
        skip_d_1 = self.skip_tranform_depth(skip_d_1)
        depth_decoder_out_2 = depth_decoder_out_2+skip_d_1
        if verbose: print("fuse after deconv4: ", depth_decoder_out_2.shape) # (144, 256)

        depth_decoder_out_1 = self.deconv1_depth(depth_decoder_out_2)
        if verbose: print("fuse after deconv5: ", depth_decoder_out_2.shape) # (288, 512)





        # RGB Encoder

        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.shape after conv1: ", rgb.shape) # (144, 256)
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.shape after bn1: ", rgb.shape)  # (144, 256)
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.shape after relu: ", rgb.shape)  # (144, 256)
        skip_r_1 = rgb

        ######################################################################

        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose: print("rgb.shape after maxpool: ", rgb.shape) # (72, 128)
        rgb = self.encoder_rgb_layer1(rgb)
        if verbose: print("rgb.shape after layer1: ", rgb.shape) # (72, 128)
        skip_r_2 = rgb

        ######################################################################
 
        rgb = self.encoder_rgb_layer2(rgb)
        if verbose: print("rgb.shape after layer2: ", rgb.shape) # (36, 64)
        skip_r_3 = rgb
        ######################################################################

        rgb = self.encoder_rgb_layer3(rgb)
        if verbose: print("rgb.shape after layer3: ", rgb.shape) # (18, 32)
        skip_r_4 = rgb
        ######################################################################

        rgb = self.encoder_rgb_layer4(rgb)
        if verbose: print("rgb.shape after layer4: ", rgb.shape) # (9, 16)

        ######################################################################

        # RGB decoder
        rgb_decoder_out_5 = self.deconv5_rgb(rgb)
        rgb_decoder_out_5 = rgb_decoder_out_5+skip_r_4
        if verbose: print("fuse after deconv1: ", rgb_decoder_out_5.shape) # (18, 32)
        rgb_decoder_out_5 = self.fusion5(rgb_decoder_out_5,depth_decoder_out_5)

        rgb_decoder_out_4 = self.deconv4_rgb(rgb_decoder_out_5)
        rgb_decoder_out_4 = rgb_decoder_out_4+skip_r_3
        if verbose: print("fuse after deconv2: ", rgb_decoder_out_4.shape) # (36, 64)
        rgb_decoder_out_4 = self.fusion4(rgb_decoder_out_4,depth_decoder_out_4)

        rgb_decoder_out_3 = self.deconv3_rgb(rgb_decoder_out_4)
        rgb_decoder_out_3 = rgb_decoder_out_3+skip_r_2
        if verbose: print("fuse after deconv3: ", rgb_decoder_out_3.shape) # (72, 128)
        rgb_seg_f3,depth_add_f3,rgb_decoder_out_3 = self.fusion3(rgb_decoder_out_3,depth_decoder_out_3)   ##  rgb image pre-segmentaion,  complement feature, fusion result 

        rgb_decoder_out_2 = self.deconv2_rgb(rgb_decoder_out_3)
        skip_r_1 = self.skip_tranform_rgb(skip_r_1)
        rgb_decoder_out_2 = rgb_decoder_out_2+skip_r_1
        if verbose: print("fuse after deconv4: ", rgb_decoder_out_2.shape) # (144, 256)
        rgb_seg_f2,depth_add_f2,rgb_decoder_out_2 = self.fusion2(rgb_decoder_out_2,depth_decoder_out_2)   ##  rgb image pre-segmentaion,  complement feature, fusion result 

        rgb_decoder_out_1 = self.deconv1_rgb(rgb_decoder_out_2)
        if verbose: print("fuse after deconv5: ", rgb_decoder_out_1.shape) # (288, 512)
        rgb_seg_f1,depth_add_f1,rgb_decoder_out_1 = self.fusion1(rgb_decoder_out_1,depth_decoder_out_1)   ##  rgb image pre-segmentaion,  complement feature, fusion result 

        return depth_decoder_out_1, rgb_decoder_out_1,rgb_seg_f1,depth_add_f1,rgb_seg_f2,depth_add_f2,rgb_seg_f3,depth_add_f3    ## depth image result, rgb image result, rgb image pre-segmentaion,  complement feature
  
class TransBottleneck(nn.Layer):

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias=False)  
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2D(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.Conv2DTranspose(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)  
        else:
            self.conv3 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  

        self.bn3 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.upsample = upsample
        self.stride = stride
 
        for m in self.modules():
            if isinstance(m, nn.Conv2D):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Conv2DTranspose):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class upbolckV1(nn.Layer):
    def __init__(self,cin,cout):
        super().__init__()
        
        self.conv1 = nn.Conv2D(cin,cin//2,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2D(cin//2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2D(cin//2,cin//2,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2D(cin//2)
        self.relu2 = nn.ReLU()       
 
        self.conv3 = nn.Conv2D(cin//2,cin//2,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2D(cin//2)
        self.relu3 = nn.ReLU()  

        self.shortcutconv = nn.Conv2D(cin,cin//2,kernel_size=1,stride=1)
        self.shortcutbn = nn.BatchNorm2D(cin//2)
        self.shortcutrelu = nn.ReLU()  

        self.se = SE_fz(in_channels=cin//2,med_channels=cin//4)

        self.transconv = nn.Conv2DTranspose(cin//2,cout,kernel_size=2, stride=2, padding=0)
        self.transbn = nn.BatchNorm2D(cout)
        self.transrelu = nn.ReLU()

    def forward(self,x):

        fusion = self.conv1(x)
        fusion = self.bn1(fusion)
        fusion = self.relu1(fusion)

        fusion = self.conv2(fusion)
        fusion = self.bn2(fusion)
        fusion = self.relu2(fusion)

        fusion = self.conv3(fusion)
        fusion = self.bn3(fusion)
        fusion = self.relu3(fusion)

        sc = self.shortcutconv(x)
        sc = self.shortcutbn(sc)
        sc = self.shortcutrelu(sc)

        fusion = fusion+sc

        fusion = self.se(fusion)


        fusion = self.transconv(fusion)
        fusion = self.transbn(fusion)
        fusion = self.transrelu(fusion)

        return fusion

class SE_fz(nn.Layer):
    def __init__(self, in_channels, med_channels):
        super(SE_fz, self).__init__()

        self.average = nn.AdaptiveAvgPool2D(1)

        self.fc1 = nn.Linear(in_channels,med_channels)
        self.bn1 = nn.BatchNorm1D(med_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(med_channels,in_channels)
        self.sg = nn.Sigmoid()
    
    def forward(self,input):
        x = input
        x = self.average(input)
        x = x.squeeze(2)
        x = x.squeeze(2)
        x = self.fc1(x)
        x= self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sg(x)
        x = x.unsqueeze(2)
        x = x.unsqueeze(3)
        out = paddle.multiply(input,x)
        return out

class FusionV10(nn.Layer):
    def __init__(self,in_channel,n_class):
        super(FusionV10, self).__init__()
        self.seg = in_channel!=n_class
        if self.seg:
            self.rgbsegconv = nn.Conv2D(in_channels=in_channel,out_channels=n_class,kernel_size=1,stride=1,padding=0)
            self.depthsegconv = nn.Conv2D(in_channels=in_channel,out_channels=n_class,kernel_size=1,stride=1,padding=0)
            self.feedback = nn.Conv2D(in_channels=n_class,out_channels=in_channel,kernel_size=1)

        self.feature_conv1 = nn.Conv2D(in_channels=n_class,out_channels=n_class,kernel_size=3,stride=1,padding=1)

        self.fusion_conv = nn.Conv2D(in_channels=3*in_channel,out_channels=in_channel,kernel_size=1)

    def forward(self,rgb,depth):

        sub_fusion_B = depth-rgb
        sub_fusion = sub_fusion_B

        if self.seg:
            rgb_seg = self.rgbsegconv(rgb)
            sub_fusion = self.depthsegconv(sub_fusion)
        else:
            rgb_seg = rgb

        depth_add = self.feature_conv1(sub_fusion)
        depth_add = depth_add+sub_fusion

        if self.seg:
            depth_add_feedback = self.feedback(depth_add)
        else:
            depth_add_feedback = depth_add

        fusion_result = paddle.concat([rgb,rgb*depth_add_feedback,depth_add_feedback],axis=1)
        fusion_result = self.fusion_conv(fusion_result)
        
        return rgb_seg,depth_add,fusion_result   ##  rgb image pre-segmentaion,  complement feature, fusion result 

class FusionV1(nn.Layer):
    def __init__(self):
        super(FusionV1, self).__init__()


    def forward(self,depth,rgb):

        fusion = depth+rgb

        return fusion

def unit_test():
    paddle.device.set_device('gpu:0')
    num_minibatch = 2
    # rgb = paddle.randn([num_minibatch, 3, 480, 640]).cuda(0)
    # thermal = paddle.randn([num_minibatch, 1, 480, 640]).cuda(0)
    rgb = paddle.randn([num_minibatch, 3, 480, 640])
    thermal = paddle.randn([num_minibatch, 1, 480, 640])
    # rtf_net = InconSeg(9).cuda(0)
    rtf_net = InconSeg(2)
    input = paddle.concat([rgb, thermal],axis=1)
    depth_decoder_out_1, rgb_decoder_out_1,rgb_seg_f1,depth_add_f1,rgb_seg_f2,depth_add_f2,rgb_seg_f3,depth_add_f3  = rtf_net(input)
    #print('The model: ', rtf_net.modules)

if __name__ == '__main__':
    unit_test()