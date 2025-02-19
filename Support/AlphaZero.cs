
using Cheese.Module;
using Godot;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

public class AlphaAI : Module<Tensor, (Tensor, Tensor)>
{
    private Module<Tensor, Tensor> _core;
    private Module<Tensor, Tensor> action;
    private Module<Tensor, Tensor> value;

    //scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95);
    public Adam optimizer;
    public optim.lr_scheduler.LRScheduler _StepLR;
    public AlphaAI() : base("AlphaAI")
    {
        var Core = new List<(string, Module<Tensor, Tensor>)>();
        var Action = new List<(string, Module<Tensor, Tensor>)>();
        var Value = new List<(string, Module<Tensor, Tensor>)>();

        Core.Add(("Conv2d_1", nn.Conv2d(7, 32, kernel_size: 3, padding: 1)));
        Core.Add(("Relu_Core_1", nn.LeakyReLU()));
        Core.Add(("Conv2d_2", nn.Conv2d(32, 64, kernel_size: 3, padding: 1)));
        Core.Add(("Relu_Core_2", nn.LeakyReLU()));
        Core.Add(("Conv2d_3", nn.Conv2d(64, 128, kernel_size: 3, padding: 1)));
        Core.Add(("Relu_Core_3", nn.LeakyReLU()));
        Core.Add(("Conv2d_4", nn.Conv2d(128, 196, kernel_size: 3, padding: 1)));
        Core.Add(("Relu_Core_4", nn.LeakyReLU()));

        Action.Add(("Conv2d_4", nn.Conv2d(196, 4, kernel_size: 1)));
        Action.Add(("Flatten", nn.Flatten()));
        Action.Add(("LinearVII", nn.Linear(324, 81)));
        Action.Add(("softmax", nn.LogSoftmax(1)));
        

        Value.Add(("Conv2d_4", nn.Conv2d(196, 2, kernel_size: 1)));
        Value.Add(("Flatten", nn.Flatten()));
        Value.Add(("LinearValue1", nn.Linear(162, 64)));
        Value.Add(("Relu", nn.LeakyReLU()));
        Value.Add(("LinearValue2", nn.Linear(64, 1)));
        Value.Add(("tanh", nn.Tanh()));

        _core = Sequential(Core);
        action = Sequential(Action);
        value = Sequential(Value);

        RegisterComponents();

        if (torch.cuda.is_available()) this.to("cuda");
    }

    public override (Tensor, Tensor) forward(Tensor input)
    {
        using var core_output = _core.forward(input.cuda());
        using var action_output = action.forward(core_output);
        using var value_output = value.forward(core_output);

        return (action_output.cpu(), value_output.cpu());
    }
}
public static class AlphaGo
{
    public static ResRollOutAI rollOutAI { get; set; }

    private static DataLoader dataloader { get; set; } = new DataLoader();
    public static ResNet alphaAI { get; set; }
    public const int SIZE = 9;
    public static Adam optimizer;
    
    public static (Tensor, Tensor) PolicyForward(Env env)
    {
        Tensor Reshape_Input, output, value;
        EnvForward(env, out Reshape_Input, out output, out value);
        output = output.reshape(new long[] { SIZE, SIZE });
        output[TensorIndex.Tensor(Reshape_Input[0, .., ..] == 1)] = -10;
        output[TensorIndex.Tensor(Reshape_Input[1, .., ..] == 1)] = -10;
        return (output.alias(), value.alias());
    }

    private static void EnvForward(Env env, out Tensor Reshape_Input, out Tensor output, out Tensor value)
    {
        Tensor All_Reshape_Input;
        EnvMakeForwardTensor(env, out Reshape_Input, out All_Reshape_Input);
        (output, value) = AlphaGo.alphaAI.forward(All_Reshape_Input);
    }

    public static void EnvMakeForwardTensor(Env env, out Tensor Reshape_Input, out Tensor All_Reshape_Input)
    {
        All_Reshape_Input = torch.zeros(new long[] { 1, 7, SIZE, SIZE });
        Tensor input = env.ToTensor();
        Reshape_Input = input.reshape(new long[] { 2, SIZE, SIZE });

        Tensor Last1_input = torch.zeros(new long[] { 2, SIZE, SIZE });
        if (env.Parent != null)
        {
            Last1_input = env.Parent.ToTensor().reshape(new long[] { 2, SIZE, SIZE });
        }
        Tensor Last2_input = torch.zeros(new long[] { 2, SIZE, SIZE });

        if (env.Parent != null && env.Parent.Parent != null)
        {
            Last2_input = env.Parent.Parent.ToTensor().reshape(new long[] { 2, SIZE, SIZE });
        }
        All_Reshape_Input[0, 0, .., ..] = input[0, .., ..];
        All_Reshape_Input[0, 1, .., ..] = Last1_input[0, .., ..];
        All_Reshape_Input[0, 2, .., ..] = Last2_input[0, .., ..];

        All_Reshape_Input[0, 3, .., ..] = input[1, .., ..];
        All_Reshape_Input[0, 4, .., ..] = Last1_input[1, .., ..];
        All_Reshape_Input[0, 5, .., ..] = Last2_input[1, .., ..];

        All_Reshape_Input[0, 6, .., ..] = env.Player;
    }

    public static (Tensor, Tensor) RotEnvTensor(Env env)
    {
        int BatchSize = 8;
        Tensor Reshape_Input;

        Tensor All_Reshape_Input;

        EnvMakeForwardTensor(env, out Reshape_Input, out All_Reshape_Input);
        
        Tensor tensors = torch.zeros(new long[] { BatchSize, All_Reshape_Input.shape[1], SIZE, SIZE });
        tensors[0, .., .., ..] = All_Reshape_Input.clone();
        tensors[1, .., .., ..] = All_Reshape_Input.rot90(1, dims: (2, 3)).clone();
        tensors[2, .., .., ..] = All_Reshape_Input.rot90(2, dims: (2, 3)).clone();
        tensors[3, .., .., ..] = All_Reshape_Input.rot90(3, dims: (2, 3)).clone();

        tensors[4, .., .., ..] = All_Reshape_Input.flip(new long[] { 2 }).rot90(0, dims: (2, 3)).clone();
        tensors[5, .., .., ..] = All_Reshape_Input.flip(new long[] { 2 }).rot90(1, dims: (2, 3)).clone();
        tensors[6, .., .., ..] = All_Reshape_Input.flip(new long[] { 2 }).rot90(2, dims: (2, 3)).clone();
        tensors[7, .., .., ..] = All_Reshape_Input.flip(new long[] { 2 }).rot90(3, dims: (2, 3)).clone();

        //tensors[7, .., .., ..] = All_Reshape_Input.flip(new long[] { 2 }).rot90(3, dims: (2, 3)).clone();

        Tensor targets = torch.zeros(new long[] { BatchSize, SIZE * SIZE });

        targets[0, ..] = env.TensorToLearn.flatten().clone();
        targets[1, ..] = env.TensorToLearn.rot90(1).flatten().clone();
        targets[2, ..] = env.TensorToLearn.rot90(2).flatten().clone();
        targets[3, ..] = env.TensorToLearn.rot90(3).flatten().clone();

        targets[4, ..] = env.TensorToLearn.flip(new long[] { 0 }).rot90(0).flatten().clone();
        targets[5, ..] = env.TensorToLearn.flip(new long[] { 0 }).rot90(1).flatten().clone();
        targets[6, ..] = env.TensorToLearn.flip(new long[] { 0 }).rot90(2).flatten().clone();
        targets[7, ..] = env.TensorToLearn.flip(new long[] { 0 }).rot90(3).flatten().clone();
        //targets[7, ..] = env.TensorToLearn.flip(new long[] { 0 }).rot90(3).flatten().clone();

        return (tensors.alias(), targets.alias());
    }
    public static void SelfPlay()
    {
        int n = 0;
        Env env = new Env();
        

        for (int i = 0; i < 81; i++)
        {
            if (env.IsEnd().Item2 != 2) break;

            OutPutText.strings.Enqueue(n.ToString() + " ");
            n++;

            Node node = new();
            MCTS mCTS = new MCTS(AlphaGo.alphaAI);
            Tensor ActProbs = mCTS.GetNextAction(env, node);
            env.TensorToLearn = ActProbs.detach().clone();

            using Tensor Where = argwhere(ActProbs == ActProbs.max()).type(ScalarType.Int32);

            int[] act = Where[TensorIndex.Tensor(torch.randperm(Where.size(0))[0])].data<int>().ToArray();
            OutPutText.strings.Enqueue(act.Join() + "\n");

            env = env.Step(act);
            
        }
        OutPutText.strings.Enqueue("\n\nfinish\n");
        OutPutText.strings.Enqueue(env.IsEnd().Item1.Join());
        OutPutText.strings.Enqueue("\nWinner=" + env.IsEnd().Item2.ToString());
        OutPutText.Env = env.Clone();

        //env.TensorToLearn = env.Parent.TensorToLearn.clone();

        DataLoader.Train(env);
        
        GC.Collect();
    }
    public static void RolloutPlay()
    {
        int n = 0;
        Env env = new Env();

        for (int i = 0; i < 81; i++)
        {
            if (env.IsEnd().Item2 != 2) break;

            OutPutText.strings.Enqueue(n.ToString() + " ");
            n++;
            AlphaGo.rollOutAI.eval();
            RollOutMCTS mCTS = new(AlphaGo.rollOutAI);
            Tensor ActProbs = mCTS.GetNextAction(env);
            env.TensorToLearn = ActProbs.detach().clone();

            using Tensor Where = argwhere(ActProbs == ActProbs.max()).type(ScalarType.Int32);

            int[] act = Where[TensorIndex.Tensor(torch.randperm(Where.size(0))[0])].data<int>().ToArray();
            OutPutText.strings.Enqueue(act.Join() + "\n");

            env = env.Step(act);
        }

        OutPutText.strings.Enqueue("\n\nfinish\n");
        OutPutText.strings.Enqueue(env.IsEnd().Item1.Join());
        OutPutText.strings.Enqueue("\nWinner=" + env.IsEnd().Item2.ToString());
        OutPutText.Env = env.Clone();

        
        DataLoader.RollTrain(env);
        DataLoader.Train(env);
        GC.Collect();
    }
}

public class DataLoader
{
    Queue<Env> ReplayData = new();
    int StepNum = 0;
    const int batchSize = 1;
    public DataLoader() { }

    public void PushIn(Env env)
    {
        ReplayData.Enqueue(env);
    }
    public void Step()
    {
        StepNum++;
        if(StepNum % batchSize == 0 && ReplayData.Count>= batchSize)
        {
            while(ReplayData.Count != 0) { Train(ReplayData.Dequeue()); }
            StepNum = 0;
        }
    }
    public static void RollTrain(Env env)
    {
        List<float> Loss1List = new();

        float entropy = 0;
        int Winner = env.IsEnd().Item2;
        if (Winner == 2) { return; }

        AlphaGo.rollOutAI.train();
        torch.set_grad_enabled(true);

        Env newenv = env;

        while (newenv.Parent != null)
        {
            if (newenv.TensorToLearn is null) { newenv = newenv.Parent; continue; }

            Tensor Inputs, Targets;
            (Inputs, Targets) = AlphaGo.RotEnvTensor(newenv);

            Tensor OutPut = AlphaGo.rollOutAI.forward(Inputs.to(CUDA)).to(CPU);

            Tensor Loss1 = -torch.sum(Targets * OutPut,1).mean();

            Tensor TotalLoss = Loss1;
            Loss1List.Add(Loss1.item<float>());

            AlphaGo.rollOutAI.adam.zero_grad();
            TotalLoss.backward();
            AlphaGo.rollOutAI.adam.step();


            entropy += torch.sum(OutPut * OutPut.exp(), 1).mean().item<float>();

            newenv = newenv.Parent;
        }

        OutPutText.Loss = (Loss1List.Sum() / Loss1List.LongCount()).ToString();

        OutPutText.Loss += "\n" + (-entropy / Loss1List.LongCount()).ToString();

    }

    public static void Train(Env env)
    {
        List<float> Loss1List = new();
        List<float> Loss2List = new();
        float entropy = 0;
        int Winner = env.IsEnd().Item2;
        if (Winner == 2) { return; }
        AlphaGo.alphaAI.train();
        torch.set_grad_enabled(true);

        Env newenv = env;


        while (newenv.Parent != null)
        {
            if (newenv.TensorToLearn is null) { newenv = newenv.Parent; continue; }

            Tensor Inputs, Targets;
            (Inputs, Targets) = AlphaGo.RotEnvTensor(newenv);

            (Tensor OutPut, Tensor Values) = AlphaGo.alphaAI.forward(Inputs.to(CUDA));
            
            OutPut = OutPut.to(CPU);
            Values = Values.to(CPU);

            Tensor Loss1 = -torch.sum(OutPut * Targets,1).mean();
            //-torch.sum(Targets * OutPut, 1).mean();

            float target = Winner == newenv.Player ? 0.999f : (Winner == 2) ? -0.0f : -0.999f;
            Tensor Loss2 = torch.nn.functional.mse_loss(Values, target);

            Tensor TotalLoss = Loss1 + Loss2;
            Loss1List.Add(Loss1.item<float>());
            Loss2List.Add(Loss2.item<float>());

            AlphaGo.alphaAI.optimizer.zero_grad();
            TotalLoss.backward();
            //float Grad = AlphaGo.alphaAI.named_parameters().ToArray()[16].parameter.grad.abs().mean().item<float>();
            AlphaGo.alphaAI.optimizer.step();

            entropy += torch.sum(OutPut * OutPut.exp(), 1).mean().item<float>(); ;

            newenv = newenv.Parent;
        }

        OutPutText.Loss = (Loss1List.Sum() / Loss1List.LongCount()).ToString() + " + " + (Loss2List.Sum() / Loss2List.Count()).ToString();

        OutPutText.Loss += "\n" + (-entropy / Loss1List.LongCount()).ToString();
        
    }
}