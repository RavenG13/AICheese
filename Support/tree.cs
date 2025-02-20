using System;
using TorchSharp;
using static TorchSharp.torch;


public class Node
{
    public Node Parent { get; private set; }
    //public int[] Act;
    public Node[,] Children;
    private int _visitCount;
    private double _valueSum;
    public double PriorP
    {
        get; set;
    }


    public Node(Node parent = null, double priorP = 1.0)
    {
        Parent = parent;
        PriorP = priorP;
        _visitCount = 0;
    }

    public double Value => _visitCount == 0 ? 0 : _valueSum / _visitCount;

    private void Update(double value)
    {
        _visitCount++;
        _valueSum += value;
    }

    /// <summary>
    /// 更新自己叶子价值和父叶子价值
    /// </summary>
    /// <param name="leafValue"></param>
    /// <param name="negate"></param>
    public void UpdateRecursive(double leafValue, bool negate = true)
    {
        Update(leafValue);
        if (IsRoot()) return;
        Parent.UpdateRecursive(negate ? -leafValue : leafValue, negate);
    }

    public bool IsLeaf() => Children is null;
    public bool IsRoot() => Parent == null;
    public int VisitCount => _visitCount;
    public Node step(int[] pos)
    {
        return Children[pos[0], pos[1]];
    }
}
public class MCTS
{
    static float _pbCBase = 1.0f;

    private nn.Module<Tensor, (Tensor, Tensor)> module;

    public MCTS(nn.Module<Tensor, (Tensor, Tensor)> module)
    {
        this.module = module;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="node"></param>
    /// <param name="alphaGo"></param>
    /// <param name="env"></param>
    /// <returns>叶节点价值</returns>
    public float ExpandLeafNode(Node node, Env env)
    {
        float LeafValue = 0;
        float[] ActionProbsArray;

        AlphaGo.EnvMakeForwardTensor(env, out Tensor input, out Tensor all_Reshape_Input);

        (Tensor ActionProbs, Tensor leaf_value) = SelfForward(all_Reshape_Input);

        ActionProbs = ActionProbs.exp();

        ActionProbsArray = ActionProbs.data<float>().ToArray();

        LeafValue = leaf_value.item<float>();

        if (env.IsEnd().Item2 != 2) return LeafValue;

        MakeRandom(env, ActionProbsArray);
        node.Children = new Node[Global.SIZE, Global.SIZE];
        for (int i = 0; i < Global.SIZE; i++)
        {
            for (int j = 0; j < Global.SIZE; j++)
            {
                node.Children[i, j] = new Node(node, ActionProbsArray[i * Global.SIZE + j]);
            }
        }
        return LeafValue;
    }
    protected virtual (Tensor Act, Tensor LeafValue) SelfForward(Tensor all_Reshape_Input)
    {
        return module.forward(all_Reshape_Input);
    }
    private static void MakeRandom(Env env, float[] ActionProbsArray)
    {
        Random random = new Random();
        (int[], double) Score = (new int[] { 0, 0 }, -1);
        for (int i = 0; i < Global.SIZE; i++)
        {
            for (int j = 0; j < Global.SIZE; j++)
            {
                if (env.ToGameState().HasPiece(i, j))
                {
                    ActionProbsArray[i * Global.SIZE + j] = -10;
                    continue;
                }
                double value = random.NextDouble();
                if (value > Score.Item2) { Score = (new int[] { i, j }, value); }
            }
        }

        ActionProbsArray[Global.SIZE * Score.Item1[0] + Score.Item1[1]] += 0.3f;
    }
    public void Simulate(Node node, Env env)
    {
        Node root = node;
        Node Leaf = node;
        Env envCopy = env.Clone();
        while (!Leaf.IsLeaf())
        {
            (int[] Act, Leaf) = SelectChild(Leaf);
            if (Act is null) break;
            envCopy = envCopy.Step(Act);
        }
        (int[] pos, byte Winner) = envCopy.IsEnd();
        bool IsDone = Winner != 2;
        float LeafValue;

        if (!IsDone) { LeafValue = ExpandLeafNode(Leaf, envCopy); }
        else
        {
            LeafValue = envCopy.Player == envCopy.IsEnd().Item2 ? 1 : -1;
        }
        Leaf.UpdateRecursive(-LeafValue);
    }
    /// <summary>
    /// 计算子节点的UCB值
    /// </summary>
    /// <param name="parent"></param>
    /// <param name="Child"></param>
    /// <returns></returns>
    public static float UcbScore(Node parent, Node Child)
    {
        double pbC = _pbCBase * Child.PriorP * Math.Sqrt(parent.VisitCount) / (Child.VisitCount + 1);
        return (float)(pbC + Child.Value);
    }
    public static (int[], Node) SelectChild(Node node)
    {
        float BestScore = -999;
        int[] action = null;
        Node Child = null;
        for (int i = 0; i < Global.SIZE; i++)
        {
            for (int j = 0; j < Global.SIZE; j++)
            {
                float Ucb = UcbScore(node, node.Children[i, j]);
                if (Ucb > BestScore)
                {
                    action = new int[] { i, j };
                    Child = node.Children[i, j];
                    BestScore = Ucb;
                }
            }
        }
        if (Child == null) Child = node;
        return (action, Child);
    }

    public Tensor GetNextAction(Env env, Node root)
    {
        AlphaGo.alphaAI.eval();

        torch.set_grad_enabled(false);
        const int NumSimulations = 400;
        int[] action = new int[2];

        if (root.Children is null) ExpandLeafNode(root, env);
        for (int i = root.VisitCount; i < NumSimulations; i++)
        {
            Simulate(root, env);
        }

        torch.Tensor ActionProbs = torch.zeros(new long[] { Global.SIZE, Global.SIZE });
        for (int i = 0; i < Global.SIZE; i++)
        {
            for (int j = 0; j < Global.SIZE; j++)
            {
                ActionProbs[i, j] = (float)root.Children[i, j].VisitCount / (float)root.VisitCount;
            }
        }

        return ActionProbs.alias();
    }

}
public class Env
{
    public Env Parent { get; private set; }
    public Tensor TensorToLearn;
    private GameState gameState;
    public int Player { get; private set; }
    public Env() { gameState = new GameState(); Player = 0; }
    public Env(GameState gameState, int player, Env parent) { this.gameState = gameState; Player = player; Parent = parent; }
    public Env Clone()
    {
        Env env = new Env(gameState.Clone(), Player, this);
        env.Parent = this.Parent;
        return env;
    }
    public Env Step(int[] Action)
    {
        GameState NextGame = gameState.Clone();
        NextGame.Place(Player, Action);
        int player = Player == 0 ? 1 : 0;
        return new Env(NextGame, player, this);
    }
    public Tensor ToTensor()
    {
        return gameState.ToTensor();
    }
    public (int[], byte) IsEnd()
    {
        return gameState.IsEnd();
    }
    public override string ToString()
    {
        string i = gameState.Show();
        return i;
    }
    public Env GetRoot()
    {
        Env root = this;
        while (root.Parent != null)
        {
            root = root.Parent;
        }
        return root;
    }
    public GameState ToGameState()
    {
        return this.gameState;
    }
}
public class PureRollOutMcts : RollOutMCTS
{
    public PureRollOutMcts() : base(null, 3600)
    {

    }
    protected override (Tensor Act, Tensor LeafValue) SelfForward(Tensor all_Reshape_Input)
    {
        return (torch.nn.functional.log_softmax(torch.rand(new long[] { 1, Global.SIZE }), 1), torch.zeros(1));
    }



}

public class RollOutMCTS : MCTS
{
    protected readonly int RollOutTimes;
    private readonly nn.Module<Tensor, Tensor> RollAI;
    public RollOutMCTS(nn.Module<Tensor, Tensor> RollAI, int RollOutTimes = 800) : base(null)
    {
        this.RollAI = RollAI;
        this.RollOutTimes = RollOutTimes;
    }

    public void RoolOut(Node root, Env env)
    {
        Random random = new Random();
        Env env1 = env.Clone();
        for (int i = 0; i < Global.SIZE * Global.SIZE; i++)
        {
            if (env1.IsEnd().Item2 != 2)
            { break; }
            double MaxRandom = -1;
            int[] Pos = new int[2];
            for (int x = 0; x < GameState.SIZE; x++)
            {
                for (int y = 0; y < GameState.SIZE; y++)
                {
                    if (env1.ToGameState().HasPiece(x, y)) { continue; }
                    double Value = random.NextDouble();
                    if (Value > MaxRandom) { Pos[0] = x; Pos[1] = y; MaxRandom = Value; }
                }
            }
            env1 = env1.Step(Pos);
        }
        byte Winner = env1.IsEnd().Item2;
        float LeafValue = Winner == env.Player ? 1f : -1f;
        root.UpdateRecursive(-LeafValue);
    }
    protected override (Tensor Act, Tensor LeafValue) SelfForward(Tensor all_Reshape_Input)
    {
        Tensor tensor = RollAI.forward(all_Reshape_Input.to(CUDA)).to(CPU);
        return (tensor, torch.zeros(1));
    }

    public Tensor GetNextAction(Env env)
    {
        Node root = new Node();
        torch.set_grad_enabled(false);
        ExpandLeafNode(root, env);

        for (int i = 0; i < RollOutTimes; i++)
        {
            Node Leaf = root;
            Env envCopy = env.Clone();
            int[] Act = new int[] { -1, -1 };
            while (!Leaf.IsLeaf())
            {
                (Act, Leaf) = SelectChild(Leaf);
                if (Act is null) break;
                envCopy = envCopy.Step(Act);
            }

            if (Leaf.VisitCount >= 20 && Leaf.IsLeaf() && envCopy.IsEnd().Item2 == 2)
            {
                ExpandLeafNode(Leaf, envCopy);
                continue;
            }
            if (Act[0] == -1)
            {
                throw new Exception("Index Wrong");
            }
            RoolOut(Leaf, envCopy);
        }
        torch.Tensor ActionProbs = torch.zeros(new long[] { Global.SIZE, Global.SIZE });
        for (int i = 0; i < Global.SIZE; i++)
        {
            for (int j = 0; j < Global.SIZE; j++)
            {
                ActionProbs[i, j] = (float)root.Children[i, j].VisitCount / (float)root.VisitCount;
            }
        }

        return ActionProbs.alias();
    }
}