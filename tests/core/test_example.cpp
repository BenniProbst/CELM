#include <gtest/gtest.h>

// Example test to verify the test framework is working
TEST(ExampleTest, BasicAssertions) {
    EXPECT_EQ(1 + 1, 2);
    EXPECT_TRUE(true);
    EXPECT_FALSE(false);
}

TEST(ExampleTest, StringComparison) {
    std::string hello = "Hello";
    std::string world = "World";
    EXPECT_NE(hello, world);
    EXPECT_EQ(hello, "Hello");
}
