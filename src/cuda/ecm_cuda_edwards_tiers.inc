// Repeated inclusion: define ECM_EDWARDS_TIER(W,T); optional ECM_EDWARDS_GROUP.
// Ascending containers, one spare bit. The 17408 tier preserves the Edwards ceiling.

#if !defined(ECM_EDWARDS_GROUP) || ECM_EDWARDS_GROUP == 0
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_128)
ECM_EDWARDS_TIER(128,4)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_192)
ECM_EDWARDS_TIER(192,4)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_256)
ECM_EDWARDS_TIER(256,4)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_384)
ECM_EDWARDS_TIER(384,4)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_512) || defined(ECM_TIER_small)
ECM_EDWARDS_TIER(512,4)
#endif
#endif

#if !defined(ECM_EDWARDS_GROUP) || ECM_EDWARDS_GROUP == 1
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_768)
ECM_EDWARDS_TIER(768,8)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1024) || defined(ECM_TIER_medium)
ECM_EDWARDS_TIER(1024,8)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1280)
ECM_EDWARDS_TIER(1280,8)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1536)
ECM_EDWARDS_TIER(1536,8)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_1792)
ECM_EDWARDS_TIER(1792,8)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_2048)
ECM_EDWARDS_TIER(2048,8)
#endif
#endif

#if !defined(ECM_EDWARDS_GROUP) || ECM_EDWARDS_GROUP == 2
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_2560)
ECM_EDWARDS_TIER(2560,16)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_3072)
ECM_EDWARDS_TIER(3072,16)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_3584)
ECM_EDWARDS_TIER(3584,16)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4096)
ECM_EDWARDS_TIER(4096,16)
#endif
#endif

#if !defined(ECM_EDWARDS_GROUP) || ECM_EDWARDS_GROUP == 3
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_4608)
ECM_EDWARDS_TIER(4608,16)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_5120)
ECM_EDWARDS_TIER(5120,16)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_5632)
ECM_EDWARDS_TIER(5632,16)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_6144)
ECM_EDWARDS_TIER(6144,16)
#endif
#endif

#if !defined(ECM_EDWARDS_GROUP) || ECM_EDWARDS_GROUP == 4
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_6656)
ECM_EDWARDS_TIER(6656,16)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_7168)
ECM_EDWARDS_TIER(7168,16)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_7680)
ECM_EDWARDS_TIER(7680,16)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_8192)
ECM_EDWARDS_TIER(8192,16)
#endif
#endif

#if !defined(ECM_EDWARDS_GROUP) || ECM_EDWARDS_GROUP == 5
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_9216)
ECM_EDWARDS_TIER(9216,32)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_10240)
ECM_EDWARDS_TIER(10240,32)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_11264)
ECM_EDWARDS_TIER(11264,32)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_12288)
ECM_EDWARDS_TIER(12288,32)
#endif
#endif

#if !defined(ECM_EDWARDS_GROUP) || ECM_EDWARDS_GROUP == 6
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_13312)
ECM_EDWARDS_TIER(13312,32)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_14336)
ECM_EDWARDS_TIER(14336,32)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_15360)
ECM_EDWARDS_TIER(15360,32)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_16384)
ECM_EDWARDS_TIER(16384,32)
#endif
#if !defined(ECM_TIERS_RESTRICTED) || defined(ECM_TIER_17408)
ECM_EDWARDS_TIER(17408,32)
#endif
#endif
