package main

import "math"

// float32ToFloat16 converts a slice of float32 values into IEEE 754-2008 binary16
// representation stored in dst. dst must be at least len(src).
func float32ToFloat16(dst []uint16, src []float32) {
	for i, v := range src {
		dst[i] = float32ToFloat16Bits(v)
	}
}

// float16ToFloat32 expands IEEE 754-2008 binary16 data into float32 values.
// dst must be at least len(src).
func float16ToFloat32(dst []float32, src []uint16) {
	for i, v := range src {
		dst[i] = float16BitsToFloat32(v)
	}
}

func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff)
	mant := bits & 0x7fffff

	switch exp {
	case 0xff:
		// Preserve NaN payloads where possible.
		if mant == 0 {
			return sign | 0x7c00
		}
		mant >>= 13
		if mant == 0 {
			mant = 1
		}
		return sign | 0x7c00 | uint16(mant)
	case 0:
		if mant == 0 {
			return sign
		}
	}

	expHalf := exp - 127 + 15
	if expHalf >= 0x1f {
		return sign | 0x7c00
	}
	mant32 := mant
	if expHalf <= 0 {
		if expHalf < -10 {
			return sign
		}
		mant32 |= 0x800000
		shift := uint(1 - expHalf)
		mant32 >>= shift
		mant32 += 0x00001000
		return sign | uint16(mant32>>13)
	}

	mant32 += 0x00001000
	if mant32&0x00800000 != 0 {
		mant32 = 0
		expHalf++
		if expHalf >= 0x1f {
			return sign | 0x7c00
		}
	}
	return sign | uint16(expHalf<<10) | uint16(mant32>>13)
}

func float16BitsToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := int((h >> 10) & 0x1f)
	mant := uint32(h & 0x3ff)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		exp = -14
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3ff
		bits := sign | uint32((exp+127)<<23) | (mant << 13)
		return math.Float32frombits(bits)
	case 0x1f:
		bits := sign | 0x7f800000 | (mant << 13)
		if mant != 0 {
			bits |= 1
		}
		return math.Float32frombits(bits)
	default:
		exp = exp - 15 + 127
		bits := sign | uint32(exp<<23) | (mant << 13)
		return math.Float32frombits(bits)
	}
}
