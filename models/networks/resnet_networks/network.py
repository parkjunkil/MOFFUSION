import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv3d(in_planes, out_planes, kernel_size=5, stride=stride,
			padding=2, padding_mode='circular', groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv3d(in_planes, out_planes, kernel_size=1, padding_mode='circular', stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				base_width=64, dilation=1, norm_layer=None):
		super(BasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm3d
		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	# Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
	# while original implementation places the stride at the first 1x1 convolution(self.conv1)
	# according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
	# This variant is also known as ResNet V1.5 and improves accuracy according to
	# https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

	expansion = 4
	
	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
			base_width=64, dilation=1, norm_layer=None):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm3d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, in_channel, kernel, padding, topo_dim, out_node1_dim, out_node2_dim, out_edge_dim, zero_init_residual=False,
				groups=1, width_per_group=64, replace_stride_with_dilation=None,
				norm_layer=None):
		super(ResNet, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm3d
		self._norm_layer = norm_layer

		self.in_channel = in_channel
		self.kernel = kernel
		self.padding = padding

		self.inplanes = 8
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None "
							"or a 3-element tuple, got {}".format(replace_stride_with_dilation))
		self.groups = groups
		self.base_width = width_per_group
		self.conv1 = nn.Conv3d(self.in_channel, self.inplanes, kernel_size=self.kernel, stride=1, padding=self.padding, padding_mode='circular',bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool3d(kernel_size=self.kernel, stride=2, padding=self.padding)
		
		
		self.layer1 = self._make_layer(block, 16, layers[0])
		self.layer2 = self._make_layer(block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
		#self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
		#self.layer4 = self._make_layer(block, 64, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
		self.avgpool = nn.AdaptiveAvgPool3d((4, 4, 4))
		#self.fc = nn.Linear(32 * 4 * 4 * 4, 32 * 4 * 4 * 4)


		##### for topo #####

		self.topo_emb = nn.Embedding(topo_dim, 32*4*4*4)

		#self.fc_topo = nn.Linear(32*4*4*4, out_topo_dim)
		self.fc_node1 = nn.Linear(64*4*4*4, out_node1_dim)
		self.fc_node2 = nn.Linear(64*4*4*4, out_node2_dim)
		self.fc_edge = nn.Linear(64*4*4*4, out_edge_dim)

		'''
		self.layer1 = self._make_layer(block, 8, layers[0])
		self.layer2 = self._make_layer(block, 16, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, 32, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
		#self.layer4 = self._make_layer(block, 64, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
		self.avgpool = nn.AdaptiveAvgPool3d((4, 4, 4))
		#self.fc = nn.Linear(32 * 4 * 4 * 4, 32 * 4 * 4 * 4)

		self.fc_topo = nn.Linear(32*4*4*4, out_topo_dim)
		self.fc_node1 = nn.Linear(32*4*4*4, out_node_dim)
		self.fc_node2 = nn.Linear(32*4*4*4, out_node_dim)
		self.fc_edge = nn.Linear(32*4*4*4, out_edge_dim)
		'''
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
				self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def _forward_impl(self, x, topo):
		# See note [TorchScript super()]
		#x_gt = x.clone()
		
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		
		x = self.layer1(x)

		x = self.layer2(x)

		#x = self.layer3(x)

		#x = self.layer4(x)

		x = self.avgpool(x)

		x = torch.flatten(x, 1)
	
		y = self.topo_emb(topo)

		#print('x', x.shape)
		#print('y', y.shape)

		x = torch.cat((x, y), dim=1)
		#print('x+y', x.shape)

		#x = self.fc(x)

		#y_topo = self.fc_topo(x)
		y_node1 = self.fc_node1(x)
		y_node2 = self.fc_node2(x)
		y_edge  = self.fc_edge(x)

		return y_node1, y_node2, y_edge

	def forward(self, x, topo):
		return self._forward_impl(x, topo)


def _resnet(arch, block, layers, in_channel, kernel, padding, topo_dim, node1_dim, node2_dim, edge_dim):
	model = ResNet(block, layers, in_channel, kernel, padding, topo_dim, node1_dim, node2_dim, edge_dim)

	return model
