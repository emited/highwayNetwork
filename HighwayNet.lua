
require 'nn'
require 'nngraph'
require 'optim'
local mnist = require 'mnist'
local disp = require 'display'

------------- OPTIONS -------------
local opt = {
	printEvery = 50,
	nbTrain = 6400,
	nbValid = 200,
	maxIter = 3000,
	optimAlgo = 'adam',
	optimState = {learningRate = 1e-2},--, weightDecay=1e-2},
	inputSize = 784,
	outputSize = 10,
	nLayers = 15,
	batchSize = 64,
	manualSeed = 123,
}
torch.manualSeed(opt.manualSeed)
--------------------------------------------


------------- HIGHWAY LAYER -------------
local function Highway(inputSize)
	local input = nn.Identity()()
	local tGateLin = nn.Linear(inputSize, inputSize)(input):annotate{name = 'tGateLin'}
	local tGate = nn.Sigmoid()(tGateLin):annotate{name = 'transform'}
	local cGate = nn.AddConstant(1)(nn.MulConstant(-1)(tGate))
	local state = nn.Linear(inputSize, inputSize)(input)
	local output = nn.CAddTable()({
		nn.CMulTable()({state, tGate}),
		nn.CMulTable()({input, cGate})})
	return nn.gModule({input}, {output})
end
--------------------------------------------


------------- NETWORK AND CRITERION -------------
function weightsInit(m)
	local name = torch.type(m)
	if name:find('Linear') then
		m.weight:normal(0.0, 1e-3)
		m.bias:fill(0)
	elseif name:find('gModule') then
		for _, node in ipairs(m.forwardnodes) do
			if node.data.annotations.name == 'tGateLin' then
				node.data.module.bias:fill(-1)
			end
		end
	elseif name:find('BatchNormalization') then
		if m.weight then m.weight:normal(1.0, 0.02) end
		if m.bias then m.bias:fill(0) end
	end
end

local net = nn.Sequential()
net:add(nn.Linear(opt.inputSize, opt.inputSize))
for n = 1, opt.nLayers-2 do
	net:add(nn.Tanh())
	net:add(nn.BatchNormalization(opt.inputSize))
	net:add(Highway(opt.inputSize))
end
net:add(nn.Tanh())
net:add(nn.BatchNormalization(opt.inputSize))
net:add(nn.Linear(opt.inputSize, opt.outputSize))
net:add(nn.SoftMax())

local params, gradParams = net:getParameters()
net:apply(weightsInit)

local crit = nn.CrossEntropyCriterion()
--------------------------------------------


------------- LOADING DATA -------------

function standardise(matrix, dim, mean, std)
	if not mean or not std then
		if not dim or dim == -1 then
			mean = matrix:mean()
			std = matrix:std()
		else
			mean = matrix:mean(dim)
			std = matrix:std(dim):add(1e-7)
		end
	end
	if torch.type(mean) == 'number' then
		return matrix:clone():add(-mean):add(1/std), mean, std
	else
		local expMean = mean:expandAs(matrix)
		local expStd = std:expandAs(matrix)
		return matrix:clone():add(-expMean):cdiv(expStd), mean, std
	end
end

local mnist_ = mnist.traindataset()
local sub_data = mnist_.data:sub(1, opt.nbTrain+opt.nbValid):double()

--trainset
local t_data = mnist_.data:sub(1, opt.nbTrain)
local t_label = mnist_.label:sub(1, opt.nbTrain)+1

train = {}
train.inputs = t_data:view(-1, opt.inputSize):double()
train.inputs, mean, std = standardise(train.inputs)
train.inputs, mean1, std1 = standardise(train.inputs, 1)
train.inputs, mean2, std2 = standardise(train.inputs, 2)
train.targets = t_label

batches = {}
batches.inputs = train.inputs:split(opt.batchSize, 1)
batches.targets = t_label:split(opt.batchSize, 1)
batches.size = #batches.inputs

-- validset
local v_data = mnist_.data:sub(1+opt.nbTrain, opt.nbTrain+opt.nbValid):double()
local v_label = mnist_.label:sub(1+opt.nbTrain, opt.nbTrain+opt.nbValid)+1
local valid = {}
valid.inputs = v_data:view(-1, opt.inputSize)
valid.inputs = standardise(valid.inputs, -1, mean, std)
valid.inputs = standardise(valid.inputs, 1, mean1, std1)
valid.inputs = standardise(valid.inputs, 2)
valid.targets = v_label
--------------------------------------------


------------- VISUALIZATION FUNCTIONS -------------

function getGradWeightValues(s)
	--returns the mean of the absolute
	--value of the weights for each
	--parameterized module

	local norms = torch.Tensor()
	local k = 1
	for i, m in ipairs(s.modules) do
		if m.gradWeight then
			norms:resize(k)
			norms[-1] = m.gradWeight:abs():mean()
			k = k + 1
		elseif torch.type(m):find('gModule') then
			for ii, mm in ipairs(m.modules) do
				if mm.gradWeight then
					norms:resize(k)
					norms[-1] = mm.gradWeight:abs():mean()
					k = k + 1
				end
			end
		end
	end
	return norms
end


local function getTransformGateOutputs(s)
	--returns the mean on the batches of the
	--output of the transform gate
	local outputs = torch.Tensor()
	local k = 1
	for i,m in ipairs(s.modules) do
		local name = torch.type(m)
		if name:find('gModule') then
			for ii, node in ipairs(m.forwardnodes) do
				if node.data.annotations.name == 'transform' then
					local output = node.data.module.output:mean(2)
					outputs:resize(k, output:size(1))
					outputs[k]:copy(output)
					k = k + 1
				end
			end
		end
	end
	return outputs
end

--------------------------------------------


------------- TRAINING NETWORK -------------

local input = torch.Tensor(opt.batchSize, opt.inputSize)
local target = torch.Tensor(opt.batchSize)
local losses = torch.Tensor()
local gradNorm = torch.Tensor()

local train_cm = optim.ConfusionMatrix({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
local valid_cm = optim.ConfusionMatrix({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})

local feval = function(x)
  if x ~= params then
     params:copy(x)
	end

	gradParams:zero()

	local _nidx_ = torch.random(batches.size)
	input:copy(batches.inputs[_nidx_])
	target:copy(batches.targets[_nidx_])
	local output = net:forward(input)
	local loss = crit:forward(output, target)
	local dl_do = crit:backward(output, target)
	net:backward(input, dl_do)
	
	return loss, gradParams
end


local plot_grad = {}
local plot_loss = {}
local plot_pred = {}
for epoch = 1, opt.maxIter do

	--gradient descent
	local _,fs = optim.sgd(feval, params, opt.optimState)

	--plotting and printing
	if (epoch-1)%opt.printEvery == 0 or epoch == opt.maxIter then

		local v_outputs = net:forward(valid.inputs):clone()
		valid_cm:zero()
		valid_cm:batchAdd(v_outputs, valid.targets)
		valid_cm:updateValids()
		disp.image(train_cm.mat, {win=23, width=200, title='ConfusionMatrix'})

		disp.image(getGradWeightValues(net):view(1, -1), {win=26, width=800, height=100, title='gradWeightValues'})
		local transGateOut = getTransformGateOutputs(net)
		transGateOut[1][1] = 0
		transGateOut[1][1] = 1
		print('transGateOutMean = '..transGateOut:mean())
		disp.image(getTransformGateOutputs(net), {win=27, title='transformGateOutputs'})


		local t_outputs = net:forward(train.inputs):clone()
		train_cm:zero()
		train_cm:batchAdd(t_outputs, train.targets)
		train_cm:updateValids()

		print('\ntraining ConfusionMatrix')
		print(train_cm)
		print('\nvalidation ConfusionMatrix')
		print(valid_cm)

		plot_loss[#plot_loss+1] = {epoch, fs[1]}
		plot_grad[#plot_grad+1] = {epoch, gradParams:norm()}
		plot_pred[#plot_pred+1] = {epoch, 1-train_cm.totalValid, 1-valid_cm.totalValid}
		disp.plot(plot_loss, {win=1, title='train loss', labels={'epoch', 'train Loss'}})
		disp.plot(plot_grad, {win=24, title='grad norm', labels={'epoch', 'gradient Norm'}})
		disp.plot(plot_pred, {win=2, title='train and test error', labels={'epoch', 'train error', 'validation error'}})
	end
end

--------------------------------------------
