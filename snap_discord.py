import discord
from discord.ext import commands
from discord import FFmpegPCMAudio

intents = discord.Intents.all()
intents.voice_states = True
client = commands.Bot(command_prefix='!', intents=intents)


