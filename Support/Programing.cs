using System;
using TorchSharp;


public class GameState
{
    public Int32[] White() 
    { 
        Int32[] white = new Int32[SIZE];
        for (int i = 0; i < SIZE; i++)
        {
            white[i] = (int)AllGame[0,i];
        }
        return white;
    }

    public Int32[] Black() 
    {
        Int32[] white = new Int32[SIZE];
        for (int i = 0; i < SIZE; i++)
        {
            white[i] = (int)AllGame[1, i];
        }
        return white;
    }

    public UInt32[,] AllGame { private set; get; } = new UInt32[2, SIZE];
    public const int SIZE = Global.SIZE;
    private static readonly torch.Tensor mask = torch.tensor(new int[] { 1 << 14, 1 << 13, 1 << 12, 1 << 11, 1 << 10, 1 << 9, 1 << 8, 1 << 7, 1 << 6, 1 << 5, 1 << 4, 1 << 3, 1 << 2, 1 << 1, 1 << 0 });
    //torch.tensor(new int[] { 1 << 14, 1 << 13, 1 << 12, 1 << 11, 1 << 10, 1 << 9, 1 << 8, 1 << 7, 1 << 6, 1 << 5, 1 << 4, 1 << 3, 1 << 2, 1 << 1, 1 << 0 });
    //torch.tensor(new int[] { 1 << 8, 1 << 7, 1 << 6, 1 << 5, 1 << 4, 1 << 3, 1 << 2, 1 << 1, 1 << 0 });
    public GameState() { }
    public GameState(uint[,] Game)
    {
        AllGame = Game;
    }
    public void Place(int player, int[] Position)
    {
        int i = Position[0];
        int j = SIZE - Position[1] - 1;

        AllGame[player, i] |= (uint)1 << j;
    }
    public bool HasPiece(int player, int[] Position)
    {
        int i = Position[0];
        int j = SIZE - Position[1] - 1;

        return (AllGame[player, i] & 1 << j) != 0;
    }
    public bool HasPiece(int i, int j)
    {
        int y = SIZE - j - 1;
        return ((AllGame[0, i] & (1 << y)) != 0 || (AllGame[1, i] & (1 << y)) != 0);
    }
    /// <summary>
    /// 0为白字，1为黑子，2为未结束
    /// </summary>
    /// <returns></returns>
    public (int[], byte) IsEnd()
    {
        for (int i = 0; i < SIZE; i++)
        {
            for (global::System.Int32 j = 0; j < SIZE; j++)
            {
                if (HasPiece(0, new int[] { i, j }))
                {
                    if (Has5(0, new int[] { i, j })) { return (new int[] { i, j }, 0); }
                }
                if (HasPiece(1, new int[] { i, j }))
                {
                    if (Has5(1, new int[] { i, j })) { return (new int[] { i, j }, 1); }
                }
            }
        }
        return (new int[] { -1, -1 }, 2);
    }
    private bool Has5(int player, int[] Position)
    {
        int i = Position[0];
        int j = Position[1];

        if (i <= SIZE - 5)
        {
            if (
                HasPiece(player, new int[] { i + 1, j }) &&
                HasPiece(player, new int[] { i + 2, j }) &&
                HasPiece(player, new int[] { i + 3, j }) &&
                HasPiece(player, new int[] { i + 4, j })
                ) { return true; }
            if (j >= 4)
            {
                if (
                HasPiece(player, new int[] { i + 1, j - 1 }) &&
                HasPiece(player, new int[] { i + 2, j - 2 }) &&
                HasPiece(player, new int[] { i + 3, j - 3 }) &&
                HasPiece(player, new int[] { i + 4, j - 4 })
                )
                { return true; }
            }
        }
        if (j <= SIZE - 5)
        {
            if (
               HasPiece(player, new int[] { i, j + 1 }) &&
               HasPiece(player, new int[] { i, j + 2 }) &&
               HasPiece(player, new int[] { i, j + 3 }) &&
               HasPiece(player, new int[] { i, j + 4 })
               ) { return true; }
            if (i <= SIZE - 5)
            {
                if (
                HasPiece(player, new int[] { i + 1, j + 1 }) &&
                HasPiece(player, new int[] { i + 2, j + 2 }) &&
                HasPiece(player, new int[] { i + 3, j + 3 }) &&
                HasPiece(player, new int[] { i + 4, j + 4 })
                )
                { return true; }

            }
        }
        return false;

    }
    public string Show()
    {
        string output = "";
        output += "White********";
        output += "\n\r";

        for(int k = 0; k < AllGame.GetLength(1); k++)
        {
            UInt32 i = (uint)AllGame[0, k];
            output += Convert.ToString(i | (1 << SIZE), 2);
            output += "\n\r";
        }
        output += "Black********";
        output += "\n\r";
        for (int k = 0; k < AllGame.GetLength(1); k++)
        {
            UInt32 i = (uint)AllGame[1, k];
            output += Convert.ToString(i | (1 << SIZE), 2);
            output += "\n\r";
        }
        return output;
    }
    public torch.Tensor ToTensor()
    {
        
        torch.Tensor output = torch.zeros(new long[] { 2, SIZE, SIZE });
        torch.Tensor j = torch.tensor(White()).reshape(SIZE, 1) & mask;
        j = j.type(torch.ScalarType.Bool);
        output[0, .., ..] = j;

        torch.Tensor i = torch.tensor(Black()).reshape(SIZE, 1) & mask;
        i = i.type(torch.ScalarType.Bool);
        output[1, .., ..] = i;
        return output.alias();
    }

    public GameState Clone()
    {
        return new GameState((uint[,])AllGame.Clone());
    }

    public override string ToString()
    {
        return base.ToString() + Show();
        
    }
}
