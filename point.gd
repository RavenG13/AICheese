extends ColorRect

signal mouse_click(n)
var have_choice := false
# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass


func _on_gui_input(event):
	if event is InputEventMouseButton and not have_choice:
		have_choice = true
		color = Color.DARK_ORCHID
		mouse_filter = 1
		
		mouse_click.emit(name)
