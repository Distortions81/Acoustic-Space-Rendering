package main

import "testing"

func TestWallClockAveragerDisabled(t *testing.T) {
	var a wallClockAverager
	a.Init(1)
	if got := a.Add(0.5); got != 0.5 {
		t.Fatalf("expected passthrough, got %v", got)
	}
}

func TestWallClockAveragerWindow(t *testing.T) {
	var a wallClockAverager
	a.Init(4)
	a.Add(1)
	a.Add(2)
	a.Add(3)
	got := a.Add(4)
	if got != 2.5 {
		t.Fatalf("expected 2.5, got %v", got)
	}
	got = a.Add(5)
	if got != 3.5 {
		t.Fatalf("expected 3.5, got %v", got)
	}
}
