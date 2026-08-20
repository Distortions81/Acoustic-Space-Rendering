module ASR

go 1.26.6

require (
	github.com/hajimehoshi/ebiten/v2 v2.9.10
	github.com/jgillich/go-opencl v0.0.0-20180608191952-a0efba3e5257
)

replace github.com/jgillich/go-opencl => ./third_party/go-opencl

require (
	github.com/ebitengine/gomobile v0.0.0-20250923094054-ea854a63cce1 // indirect
	github.com/ebitengine/hideconsole v1.0.0 // indirect
	github.com/ebitengine/oto/v3 v3.4.1 // indirect
	github.com/ebitengine/purego v0.9.0 // indirect
	github.com/jezek/xgb v1.1.1 // indirect
	golang.org/x/sync v0.21.0 // indirect
	golang.org/x/sys v0.44.0 // indirect
)
