using Cheese.Module;
using Godot;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;
using TorchSharp;
using static Godot.Control;
using static TorchSharp.torch;


public partial class node_2d : Node2D
{
    GridContainer gridContainer;
    Label Label;

    point[,] points;

    TextEdit DebugText;
    Tree tree;
    bool showText = true;
    [Export]
    int LearningTimes;
    bool Learing = false;
    SpinBox spinBox;
    Env HunmanVSAiEnv;
    bool HunmanVSAi { get; set; } = false;
    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
    {

        DebugText = GetNode<TextEdit>("DebugText");
        OutPutText.textEdit = DebugText;


        gridContainer = GetNode<GridContainer>("GridContainer");
        PackedScene packedScene = GD.Load<PackedScene>("res://point.tscn");
        spinBox = GetNode<SpinBox>("/root/Node2D/BoxContainer/SpinBox");
        points = new point[9, 9];

        for (int i = 0; i < points.GetLength(0); i++)
        {
            for (int j = 0; j < points.GetLength(1); j++)
            {
                point node = packedScene.Instantiate<point>();
                node.Name = i.ToString() + j.ToString();
                node.Color = Godot.Color.Color8(200, 200, 200);
                gridContainer.AddChild(node);
                node.MouseClick += MouseClick;
                points[i, j] = node;
            }
        }

        Label = GetNode<Label>("Label");


        AlphaGo.alphaAI = new ResNet("res", 6);
        AlphaGo.alphaAI.to(CUDA);
        AlphaGo.alphaAI = (ResNet)AlphaGo.alphaAI.load("./ModuleSave/New.dat");

        AlphaGo.rollOutAI = new("test");
        AlphaGo.rollOutAI.to(CUDA);
        AlphaGo.rollOutAI.load("./ModuleSave/ResrollOutAI.dat");

        AlphaGo.alphaAI.optimizer = new(AlphaGo.alphaAI.parameters(), lr: 1E-4);
        AlphaGo.rollOutAI.adam = new(AlphaGo.rollOutAI.parameters(), lr: 5E-5);

        //Test();

    }
    public void Test()
    {
        Env env = new Env();
        env = env.Step(new int[] { 1, 2 });
        env = env.Step(new int[] { 3, 5 });
        env = env.Step(new int[] { 1, 3 });
        env = env.Step(new int[] { 5, 8 });
        env = env.Step(new int[] { 1, 1 });
        env = env.Step(new int[] { 4, 8 });
        env = env.Step(new int[] { 1, 0 });
        //env = env.Step(new int[] { 1, 8 });
        MCTS mCTS = new MCTS(AlphaGo.alphaAI);

        Tensor ActProbs = mCTS.GetNextAction(env, new Node());
        using Tensor Where = argwhere(ActProbs == ActProbs.max()).type(ScalarType.Int32);

        int[] act = Where[TensorIndex.Tensor(torch.randperm(Where.size(0))[0])].data<int>().ToArray();
        env = env.Step(act);
        OutPutText.Env = env;
    }
    public async void Evalation()
    {
        await Task.Run(() =>
        {
            Env env = new Env();
            PureRollOutMcts pureRollOutMcts = new PureRollOutMcts();
            RollOutMCTS rollOut = new RollOutMCTS(AlphaGo.rollOutAI);
            for (int i = 0; i < 81; i++)
            {
                Tensor next;
                if (i % 2 == 0) { next = pureRollOutMcts.GetNextAction(env); }
                else { next = rollOut.GetNextAction(env); }

                using Tensor Where = argwhere(next == next.max()).type(ScalarType.Int32);

                int[] act = Where[TensorIndex.Tensor(torch.randperm(Where.size(0))[0])].data<int>().ToArray();
                OutPutText.strings.Enqueue(act.Join() + "\n");

                env = env.Step(act);
                if (env.IsEnd().Item2 != 2) { break; }
            }
            OutPutText.Env = env;
            OutPutText.strings.Enqueue("Winner=" + env.IsEnd().Item2);
        });
    }

    public void SaveModule()
    {
        AlphaGo.alphaAI.save("./ModuleSave/New.dat");
        AlphaGo.rollOutAI.save("./ModuleSave/ResrollOutAI.dat");
        OutPutText.strings.Enqueue("Save");
    }
    public void ShowText()
    {
        showText = !showText;
    }
    public void RefrashText()
    {
        DebugText.Text = "";
        OutPutText.strings.Clear();
    }
    public async void Study()
    {
        if (Learing) return;
        Learing = true;
        for (int i = 0; i < LearningTimes; i++)
        {
            await Task.Run(() => AlphaGo.SelfPlay());

            DebugText.Text += $"StudyTimes{i}\n";
        }
        Learing = false;
        //await Task.Run(() => Eval());

    }


    public void ShowMap()
    {
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                points[i, j].Color = Godot.Color.Color8(0, 250, 0);
                points[i, j].GetNode<Label>("Label").Text = "";
            }
        }
        Env root = OutPutText.Env;
        Env End = OutPutText.Env;
        int Length = 0;
        if (root is null) { return; }
        while (root.Parent != null)
        {
            root = root.Parent;
            Length++;
        }
        Env env = End;
        while (env.Parent != null)
        {
            GameState g1 = env.ToGameState();
            GameState g2 = env.Parent.ToGameState();
            for (int i = 0; i < g1.White.Length; i++)
            {
                int w = g1.White[i] ^ g2.White[i];
                if (w != 0)
                {
                    int Index = 8 - (int)System.MathF.Log2(w);
                    points[i, Index].Color = new Color(255, 255, 255);
                    points[i, Index].GetNode<Label>("Label").Text = Length.ToString() + "\n";

                    break;
                }
                int B = g1.Black[i] ^ g2.Black[i];
                if (B != 0)
                {
                    int Index = 8 - (int)System.MathF.Log2(B);
                    points[i, Index].Color = new Color(0, 0, 0);
                    points[i, Index].GetNode<Label>("Label").Text = Length.ToString() + "\n";

                    break;
                }
            }

            Length--;
            env = env.Parent;
        }
    }
    public void PlayerWithAI()
    {
        HunmanVSAi = !HunmanVSAi;


        HunmanVSAiEnv = new();
        OutPutText.Env = HunmanVSAiEnv;
        ShowMap();


        foreach (var i in points)
        {
            if (HunmanVSAi) i.MouseFilter = Control.MouseFilterEnum.Pass;
            else i.MouseFilter = Control.MouseFilterEnum.Ignore;
        }
    }
    public void MouseClick(point point)
    {
        if (!HunmanVSAi) return;

        int[] position = new int[2];
        for (int i = 0; i < points.GetLength(0); i++)
        {
            for (int j = 0; j < points.GetLength(1); j++)
            {
                if (points[i, j] == point)
                {
                    position = new int[] { i, j };
                    break;
                }
            }
        }

        OutPutText.strings.Enqueue(position[0].ToString());
        HunmanVSAiEnv = HunmanVSAiEnv.Step(position);

        RollOutMCTS rollOutMCTS = new(AlphaGo.rollOutAI, 3200);
        Tensor next = rollOutMCTS.GetNextAction(HunmanVSAiEnv);
        using Tensor Where = argwhere(next == next.max()).type(ScalarType.Int32);

        int[] act = Where[TensorIndex.Tensor(torch.randperm(Where.size(0))[0])].data<int>().ToArray();
        OutPutText.strings.Enqueue(act.Join() + "\n");

        HunmanVSAiEnv = HunmanVSAiEnv.Step(act);
        points[act[0], act[1]].MouseFilter = MouseFilterEnum.Ignore;
        if (HunmanVSAiEnv.IsEnd().Item2 != 2)
        {
            OutPutText.strings.Enqueue("Winner=" + HunmanVSAiEnv.IsEnd().Item2);
            foreach (var i in points)
            {
                i.MouseFilter = Control.MouseFilterEnum.Ignore;
            }
        }

        OutPutText.Env = HunmanVSAiEnv.Clone();
        ShowMap();

    }

    public override void _PhysicsProcess(double delta)
    {
        if (showText)
        {
            if (OutPutText.strings.TryDequeue(out string str))
            {
                DebugText.Text += str;
                if (DebugText.Text.Length > 3000) { DebugText.Text = str; }
            }
        }
        Label.Text = OutPutText.Loss;
        LearningTimes = (int)spinBox.Value;
    }
}

public static class OutPutText
{
    public static TextEdit textEdit;
    public static ConcurrentQueue<string> strings = new();
    public static Env Env;
    public static string Loss;

}
/*

    public void ReConnect()
    {

        GD.Print("renet");
        client = new MyClient();


    }

    public override void _ExitTree()
    {
        client.stream.Close();
    }
    public void ClearBuffer()
    {
        client.stream.Flush();
    }


    class MyClient
    {
        public NetworkStream? stream;
        public TcpClient tcpClient;
        public delegate void ConnectFinish();

        public MyClient()
        {
            //Uses the IP address and port number to establish a socket connection.
            tcpClient = new TcpClient();
            string hostName = Dns.GetHostName();
            IPHostEntry hostEntry = Dns.GetHostEntry(hostName);
            IPAddress[] addresses = hostEntry.AddressList;
            Connect(tcpClient, addresses);
        }
        public async void Connect(TcpClient tcpClient, IPAddress[] addresses)
        {
            await tcpClient.ConnectAsync(addresses, 8080);
            this.stream = tcpClient.GetStream();
        }

        public int[] ClintRead()
        {
            byte[] buffer = new byte[449];
            int num = 0;
            if (stream != null)
            {
                num = stream.Read(buffer);
            }
            //decode
            string result = UTF8Encoding.UTF8.GetString(buffer);

            char[] separators = new char[] { ',' };

            string[] parts = result.Split(separators);

            int[] game = new int[225];

            for (int i = 0; i < parts.Length; i++)
            {
                game[i] = int.Parse(parts[i]);
            }
            return game;
        }

        public void ClintWrite(int[] games)
        {
            string str = string.Join(",", games);
            byte[] bytes = Encoding.UTF8.GetBytes(str);
            if (stream != null)
            {
                stream.Write(bytes, 0, bytes.Length);
            }
            else
            {
                Console.WriteLine("No Stream");
            }
        }
    }
*/

