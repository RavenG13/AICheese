using Godot;
using System;

public partial class point : ColorRect
{

	[Signal]
	public delegate void MouseClickEventHandler(point point);
    private Color colorCopy;
    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
	{
		GuiInput += MouseInput;
        MouseEntered += Point_MouseEntered;
        MouseExited += Point_MouseExited;

        Color = Godot.Color.Color8(0, 250, 0);
    }

    private void Point_MouseExited()
    {
        if (MouseFilter != MouseFilterEnum.Ignore) { Color = colorCopy; }
    }

    private void Point_MouseEntered()
    {
        if (MouseFilter != MouseFilterEnum.Ignore)
        {
            colorCopy = Color;
            Color = Godot.Color.Color8(0, 0, 230);
        }
    }

    // Called every frame. 'delta' is the elapsed time since the previous frame.
    public override void _Process(double delta)
	{
	}

	public void MouseInput(InputEvent inputEvent)
	{
		if(inputEvent is InputEventMouseButton) 
		{
            InputEventMouseButton input = inputEvent as InputEventMouseButton;
            if(input.ButtonIndex == MouseButton.Left && input.Pressed == false)
			{
				EmitSignal(SignalName.MouseClick,this);

                this.MouseFilter = MouseFilterEnum.Ignore;
            }

        }
	}
}
