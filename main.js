var canvas = document.getElementsByTagName("canvas")[0];
canvas.width = 280;
canvas.height = 280;
var c = canvas.getContext('2d');

c.strokeStyle = "white";
c.lineWidth = 10;

var data = [];
var xs;
var ys;
var model;
var training = false;
var prepared = false;

async function getData() //Pobiera dane z bazy
{
    for(let i=0;i<10;i++)
    {
        await $.getJSON(`https://raw.githubusercontent.com/cazala/mnist/master/src/digits/${i}.json`, function(json) {
            data[i] = json.data;
        });
    }
    onDataGotted();
    prepareData(); //Po odebraniu danych trzeba je przygotować
}
getData(); //Wywołuje od razu.

function prepareData() //Szlifuje dane do formy tensorów
{
    let inputs = [];
    let outputs = [];
    for(let i=0;i<10;i++)
    {
        for(let j=0;j<800;j++)
        {
            inputs.push(data[i].slice(j*28*28,(j+1)*28*28));
            let output = [0,0,0,0,0,0,0,0,0,0];
            output[i] = 1;
            outputs.push(output);
        }
    }
    xs = tf.tensor(inputs);
    ys = tf.tensor(outputs);
    onDataPrepared();
}

function createModel() //Tworzy sieć tego nie dotykamy.
{
    const model = tf.sequential();

    const hidden1 = tf.layers.dense({
        units: 64,
        inputShape: 784,
        activation: 'relu6',
    });
    model.add(hidden1);
 

    const hidden2 = tf.layers.dense({
        units: 64,
        activation: 'relu6'
    });
    model.add(hidden2);

    const hidden3 = tf.layers.dense({
        units: 64,
        activation: 'relu6'
    });
    model.add(hidden3);

    const output = tf.layers.dense({
        units: 10,
        activation: 'relu6'
    });
    model.add(output);


    model.compile({
        optimizer: tf.train.sgd(0.3),
        loss: tf.losses.meanSquaredError
    });

    return model;
}

model = createModel();
training = false; //Zmienna co sprawdza, czy sieć jest aktualnie trenowana. Jeśli jest wszystkie inne akcje powinny zostać zablokowane aż do ukończenia

async function train(epochs) //Trenuje sieć przez zadaną liczbę epok.
{
    response = await model.fit(xs, ys, {
        epochs: epochs, 
        callbacks: {
            onEpochEnd: (epoch,logs)=>{onEpochEnd(epoch,logs,epochs);},
            onTrainEnd: (logs)=>{onTrainEnd();}
        }
    });
	training = false;
}

function returnConsole(text) 
{
	document.getElementById("console").innerHTML+=text;
}

var lastX;
var lastY;

var mousePressed = false; //Czy myszka jest przyciśnięta

window.addEventListener("load",_=>
{
	returnConsole('<p class="console_warning">Fetching data, please wait.</p>');
	document.getElementById("console_container").style.opacity = 1;
	canvas.addEventListener("mousedown",e=> //Przyciśnięcie myszki (tylko na canvasie)
	{
        mousePressed = true;
        lastX = e.clientX;
        lastY = e.clientY;
	});
	
	canvas.addEventListener("mousemove",e=> //Przesunięcie myszki (tylko na canvasie)
	{
		if(mousePressed && prepared)
		{
            c.beginPath();
            c.moveTo(lastX-canvas.offsetLeft,lastY-canvas.offsetTop);
            c.lineTo(e.clientX-canvas.offsetLeft,e.clientY-canvas.offsetTop);
            c.stroke();
			c.beginPath();
			let iks = (e.clientX-canvas.offsetLeft);
			let igrek = (e.clientY-canvas.offsetTop);
			c.arc(iks,igrek,5,0,Math.PI*2);
			c.fillStyle = "white";
            c.fill();
            lastX = e.clientX;
            lastY = e.clientY;
		}
	});
	
	window.addEventListener("mouseup",e=> //Puszczenie myszki (gdziekolwiek)
	{
		mousePressed = false;
	});
	
	document.getElementById("train").addEventListener("click",onTrainBegin);
	
	document.getElementById("predict").addEventListener("click",_=> //Pobiera dane z canvas i predictuje numer
	{
		var imageData = c.getImageData(0, 0, 280, 280).data;
        imageData = imageData.filter((e,i)=>i%4==1);
        var image = [];
        var imageSmol = [];
        imageData.forEach(e=>image.push(e/255));
        for(let i=0;i<28;i++)
        {
            for(let j=0;j<28;j++)
            {
                let rows = []; 
                for(let k=0;k<10;k++)
                {
                    let row = image.slice(i*2800+j*10+k*280,i*2800+(j+1)*10+k*280); //Załamanie zasady De'Lamberta
                    rows.push(row.reduce((acc,cur)=>acc+cur)/10);
                }
                imageSmol.push(rows.reduce((acc,cur)=>acc+cur)/10);
            }
        }
        var tensor = tf.tensor([imageSmol]);
        model.predict(tensor).data().then(t=>onPredicting(t.reduce((iMax,x,i,arr)=>x>arr[iMax]?i:iMax,0)));
    });
    
    function clearCanvas()
    {
        c.fillStyle = "black";
        c.fillRect(0,0,280,280);
    }

    document.getElementById("clear").addEventListener("click",clearCanvas);
	
	window.addEventListener("keydown",e=>
	{
		if(e.keyCode == 192)
		{
			var x = document.getElementById("console_container");
			x.style.opacity = 1 - parseInt(x.style.opacity);
		}
		if(e.keyCode == 67)
		{
			if(document.getElementsByClassName("console_compact")[0])
			{
				document.getElementById("console_container").classList.remove("console_compact");
			}
			else
			{
				document.getElementById("console_container").classList.add("console_compact");
			}
		}
	}, false);
	
	document.getElementById("canvas_help").addEventListener("mouseover",_=>
	{
		document.getElementById("canvas_help").style.display="none";
	});
    
    clearCanvas();
});

function onDataGotted() //Wywoływana, gdy dane zostaną pobrane z serwera (następnie są przygotowywane)
{
	returnConsole("<p>Data fetched, ready for preparation.</p>");
}

function onDataPrepared() //Wywoływana, gdy dane zostaną przygotowane (wtedy można robić rzeczy)
{
    prepared = true;
    returnConsole("<p>Data prepared.</p>");
	returnConsole('<p class="">Press c to toggle collapse mode.</p>');
	returnConsole('<p class="">Press ` to hide console.</p>');
}

function onEpochEnd(epoch, logs, epochs) //Wywoływana na końcu każdej epoki uczenia, argumenty to (numer_epoki, logi, ilosc_epok)
{
    let percent = Math.round((epoch+1)*100/epochs); //Progres w procentach
    let loss = logs.loss; //Omg, is this? Yes. To je loss. Mówi jak dobrze działa sieć (im mniejszy tym lepszy);
	document.getElementById("progress_bar").style.width = percent+"%";
	document.getElementById("progress_bar_p").innerHTML = percent+"%";
	returnConsole(`<p>Training... ${percent}% | ${loss.toFixed(5)} loss.</p>`);
}

function onPredicting(digit) //Funckja wywoływana po przewidzeniu numeru. Argument to cyfra, którą sieć przewidziała 
{
    document.getElementById("result").innerHTML = digit;
	returnConsole(`<p>Predicted: ${digit}.</p>`);
}

function onTrainBegin(event) //Po kliknięciu "train"
{
    let epochs = document.getElementById("epochInput").value;
	
	if(!parseInt(epochs) || epochs != parseInt(Math.abs(epochs)))
	{
		returnConsole('<p class="console_warning">The Epochs field must be an integer above 0.</p>');
	}
	else
	{
        epochs = parseInt(epochs);
		if(epochs>100 && event.srcElement != document.getElementById("console_button_confirm"))
		{
			returnConsole('<p class="console_warning">Warning: This is going to take a long time. Are you sure you want to do this? (You can always repeat training if you\'re not satisfied with the results, but you can\'t stop mid-training)</p><p class="console_button" id="console_button_confirm">Yes, I\'m sure.</p>');
			document.getElementById("console_button_confirm").addEventListener("click",onTrainBegin);
		}
		else
		{
			if(event.srcElement == document.getElementById("console_button_confirm"))
			{
				document.getElementById("console_button_confirm").classList.add("console_button_clicked");
			}
			returnConsole("<p>Training begins.</p>");
			document.getElementById("train").classList.add("class","button_train_disabled");
			training = true;
			train(epochs);
		}
	}
    
}

function onTrainEnd() //Gdy trenowanie zostanie zakończone
{
    document.getElementById("train").classList.remove("button_train_disabled");
    returnConsole("<p>Training done.</p>");
}
animate();


