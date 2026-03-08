const mongoose = require("mongoose")

const connectDB = async()=>{
    try {
        await mongoose.connect(
          "mongodb+srv://someshrocks144:somesh@cluster0.ezcv3ck.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        );
        console.log("Connected to DB!!")
    } catch (error) {
        console.log(error)
    }
}

module.exports = connectDB