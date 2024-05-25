package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/therecipe/qt/widgets"
)

type point struct {
	x, y float64
}

func RandomNumberGenerator() *rand.Rand {
	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)
	return r1
}

func RandArray(from, to float64, N int) (a []point) {
	rng := RandomNumberGenerator()
	a = make([]point, N)
	for i := 0; i < N; i++ {
		a[i] = point{rng.Float64()*(to-from) + from, rng.Float64()*(to-from) + from}
	}
	return
}

func task(pStart, pEnd float64, N, times int, BoolFunc func(dot point) bool, CorrectAns float64) (ans, diff float64) {
	ans = 0.0
	for i := 0; i < times; i++ {
		dots := RandArray(pStart, pEnd, N)
		correct := 0
		for j := 0; j < len(dots); j++ {
			if BoolFunc(dots[j]) {
				correct += 1
			}
		}
		ans += math.Abs(float64(correct) / float64(N) * math.Pow(pStart-pEnd, 2))
	}
	return ans / float64(times), math.Abs((ans/float64(times) - CorrectAns) / CorrectAns * 100)
}

func dotIsInFirstArea(dot point) bool {
	return (-1*math.Pow(dot.x, 3)-5*math.Pow(dot.y, 3) < 2) && (-dot.x+dot.y < 2) && (math.Abs(dot.x) < 2) && (math.Abs(dot.y) < 2)
}

func dotIsInSecondArea(dot point) bool {
	return (dot.y < math.Pow(dot.x, 1/3)) && (math.Abs(dot.x) < 8) && (math.Abs(dot.y) < 8)
}

func main() {
	// Создание приложения Qt
	app := widgets.NewQApplication(len([]string{}), []string{})

	// Создание основного окна
	window := widgets.NewQMainWindow(nil, 0)
	window.SetWindowTitle("Monte Carlo Simulation")

	// Создание центрального виджета
	centralWidget := widgets.NewQWidget(nil, 0)
	window.SetCentralWidget(centralWidget)

	// Создание компоновки
	layout := widgets.NewQVBoxLayout()

	// Создание кнопки для выполнения задачи 1
	task1Button := widgets.NewQPushButton2("Run Task 1", nil)
	task1Result := widgets.NewQLabel2("", nil, 0)

	// Обработка нажатия кнопки
	task1Button.ConnectClicked(func(checked bool) {
		ans, diff := task(-2, 2, int(math.Pow10(6)), 100, dotIsInFirstArea, 8.38467)
		resultText := fmt.Sprintf("Task 1 answer: %f\nWhile correct answer is 8.38467\nDiff: %f%%", ans, diff)
		task1Result.SetText(resultText)
	})

	// Добавление кнопки и метки в компоновку
	layout.AddWidget(task1Button, 0, 0)
	layout.AddWidget(task1Result, 0, 0)

	// Создание кнопок и меток для задач 2 с различными N
	for i := 0; i < 7; i++ {
		task2Button := widgets.NewQPushButton2(fmt.Sprintf("Run Task 2 with N = 10^%d", i), nil)
		task2Result := widgets.NewQLabel2("", nil, 0)
		N := int(math.Pow10(i))

		task2Button.ConnectClicked(func(checked bool) {
			ans, diff := task(-2, 2, N, 100, dotIsInSecondArea, 12)
			resultText := fmt.Sprintf("Task 2 answer with N = %d: %f\nWhile correct answer is 12\nDiff: %f%%", N, ans, diff)
			task2Result.SetText(resultText)
		})

		layout.AddWidget(task2Button, 0, 0)
		layout.AddWidget(task2Result, 0, 0)
	}

	// Установка компоновки центрального виджета
	centralWidget.SetLayout(layout)

	// Отображение окна
	window.Show()

	// Запуск цикла обработки событий
	app.Exec()
}
