using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Cheese.Module
{
    internal class ConvBlock : Module<Tensor, Tensor>
    {
        Module<Tensor, Tensor> _Conv2D;
        Module<Tensor, Tensor> _BatchNorm2d;
        public ConvBlock(string name, int in_channel, int out_channel, int kernel_size, int padding = 0) : base(name)
        {
            _Conv2D = nn.Conv2d(in_channel, out_channel, kernel_size, padding:padding);
            _BatchNorm2d = torch.nn.BatchNorm2d(out_channel);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            return torch.nn.functional.relu(_BatchNorm2d.forward(_Conv2D.forward(input)));
        }
    }
    internal class ResBlock : Module<Tensor, Tensor>
    {
        Module<Tensor, Tensor> _Conv2D1;
        Module<Tensor, Tensor> _Conv2D2;
        Module<Tensor, Tensor> _BatchNorm2d1;
        Module<Tensor, Tensor> _BatchNorm2d2;
        public ResBlock(string name,int in_channel,int out_channel) : base(name)
        {
            _Conv2D1 = nn.Conv2d(in_channel, out_channel, kernel_size: 3, padding: 1);
            _Conv2D2 = nn.Conv2d(in_channel, out_channel, kernel_size: 3, padding: 1);
            _BatchNorm2d1 = nn.BatchNorm2d(out_channel);
            _BatchNorm2d2 = nn.BatchNorm2d(out_channel);

            RegisterComponents();
        }


        public override Tensor forward(Tensor input)
        {
            using Tensor out1 = torch.nn.functional.relu(_BatchNorm2d1.forward(_Conv2D1.forward(input)));
            using Tensor out2 = _BatchNorm2d2.forward(_Conv2D2.forward(input));
            return torch.nn.functional.relu(input + out2);
        }
    }
    public class ResNet : Module<Tensor, (Tensor, Tensor)>
    {
        Module<Tensor, Tensor> _Core;
        Module<Tensor, Tensor> _ValueHead;
        Module<Tensor, Tensor> _PolicyHead;
        public Adam optimizer;
        public ResNet(string name,int in_channel) : base(name)
        {
            var _core = new List<(string,Module<Tensor,Tensor>)>();
            _core.Add(("InConv", new ConvBlock("Conv1", in_channel, 128, 3, 1)));
            _core.Add(("Res1", new ResBlock("Res1", 128, 128)));
            _core.Add(("Res2", new ResBlock("Res2", 128, 128)));
            _core.Add(("Res3", new ResBlock("Res3", 128, 128)));
            _Core = nn.Sequential(_core);

            var _value = new List<(string, Module<Tensor, Tensor>)>();
            _value.Add(("ValueConv1", new ConvBlock("ValueConv1", 128, 1, 1)));
            _value.Add(("Flutten", nn.Flatten(1)));
            _value.Add(("lin1", nn.Linear(81,128)));
            _value.Add(("relu", nn.ReLU()));
            _value.Add(("lin2", nn.Linear(128, 1)));
            _value.Add(("tanh", nn.Tanh()));
            _ValueHead = nn.Sequential(_value);

            var _policy = new List<(string, Module<Tensor, Tensor>)>();
            _policy.Add(("ConvBlock", new ConvBlock("ConvBlock", 128, 2, 1)));
            _policy.Add(("Flutten", nn.Flatten(1)));
            _policy.Add(("Linear", nn.Linear(162, 81)));
            _policy.Add(("Out", nn.LogSoftmax(1)));
            _PolicyHead = nn.Sequential(_policy);

            RegisterComponents();
        }

        public override (Tensor, Tensor) forward(Tensor input)
        {
            using var x = _Core.forward(input.cuda());
            return (_PolicyHead.forward(x).cpu(), _ValueHead.forward(x).cpu());
        }
    }

    public class ResRollOutAI : Module<Tensor, Tensor>
    {
        public Adam adam { get; set; }
        Module<Tensor, Tensor> _Core;
        public ResRollOutAI(string name) : base(name)
        {
            var core = new List<(string, Module<Tensor, Tensor>)>();
            core.Add(("Conv2D1", Conv2d(7, 64, 3, 1)));
            core.Add(("Relu", ReLU()));
            core.Add(("Res1", new ResBlock("Res1", 64, 64)));
            core.Add(("Res2", new ResBlock("Res2", 64, 64)));
            core.Add(("Res3", new ResBlock("Res3", 64, 64)));
            core.Add(("Filter", Conv2d(64, 4, 1)));
            core.Add(("Relu", ReLU()));
            core.Add(("Fletten", Flatten()));
            core.Add(("Linear1", Linear(196, 162)));
            core.Add(("Relu", ReLU()));
            core.Add(("Linear2", Linear(162, 81)));
            core.Add(("out", LogSoftmax(1)));

            _Core = nn.Sequential(core);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            return _Core.forward(input);
        }
    }
    
}
