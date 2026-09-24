import { describe, expect, it } from 'vitest'
import { trimTrailingZeros } from '@/utils/format'

describe('trimTrailingZeros', () => {
  it('keeps zeros that belong to the integer', () => {
    expect(trimTrailingZeros('0')).toBe('0')
    expect(trimTrailingZeros('10')).toBe('10')
    expect(trimTrailingZeros('100')).toBe('100')
  })

  it('drops only fractional trailing zeros', () => {
    expect(trimTrailingZeros('10.00')).toBe('10')
    expect(trimTrailingZeros('1.10')).toBe('1.1')
    expect(trimTrailingZeros('0.00')).toBe('0')
  })
})
