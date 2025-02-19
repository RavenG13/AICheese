extends Node2D

@export var use_ip:String = "10.98.126.13"

var peers=[]
var socket = StreamPeerTCP.new()
var game = []
var thread:Thread
var semaphore: Semaphore


func _ready():
	#socket.connect_to_host("use_ip",8080)
	peers.append(socket)
	var points = load("res://point.tscn")
	for i in range(0,225):
		var instance_point = points.instantiate()
		instance_point.name = String.num(i)
		instance_point.mouse_click.connect(on_mouse_click)
		$GridContainer.add_child(instance_point,true)
	game.resize(225)
	for k in range(0,225):
		game[k] = 0
		
	thread = Thread.new()
	thread.start(TcpReceive)
	semaphore = Semaphore.new()
	print(socket.connect_to_host("use_ip",8080))


func _process(delta):
	socket.poll()
	
	#print(socket.get_status())
	$Label.text = String.num(int(socket.get_status()))
	
func save_file(content):
	var file = FileAccess.open("file.txt", FileAccess.WRITE)
	file.store_csv_line(content,',')
	
func on_mouse_click(n):
	n=int(String(n))
	game[n] = 2
	_on_button_pressed()
	
func _on_button_pressed():
	if(socket.get_status() == 2):
		#socket.put_utf8_string("hello")
		socket.put_data(PackedByteArray(game))
		await get_tree().create_timer(0.2).timeout
		_on_button_2_pressed()
	#save_file(game)
	#var op =[]
	#print(OS.execute("powershell.exe",["-Command","f:/学习/cheese/running.py"],op))
	#print(op)
	

func _on_button_2_pressed():
	semaphore.post()
	await get_tree().create_timer(0.2).timeout
	update_game()
	pass

func update_game():
	for i in range(0,225):
		if game[i] == 1:
			$GridContainer.get_child(i).color = Color.CHARTREUSE
			$GridContainer.get_child(i).have_choice = true
		if game[i] == 2:
			$GridContainer.get_child(i).color = Color.BLUE_VIOLET
			$GridContainer.get_child(i).have_choice = true
		if game[i] == 0:
			$GridContainer.get_child(i).color = Color.WHITE
			$GridContainer.get_child(i).have_choice = false
	

func TcpReceive():
	while true:
		semaphore.wait() # Wait until posted.
		var get_in = socket.get_utf8_string()
		var splited = get_in.split(",",false)
		var result = []
		for i in splited:
			result.append(int(i))
		game = result
	pass

func _on_button_3_pressed():
	game=[]
	game.resize(225)
	for i in range(0,225):
		game[i] = 0


func 重新连接():
	print("重新连接")
	socket.connect_to_host("use_ip",8080)
	pass # Replace with function body.
